"""
Node-based manager using pure Monte Carlo Tree Search. It implements a tree-based
search strategy that allows for more flexible exploration and exploitation of solution
space. It also ensures all available tools are tried during the exploration process.
"""

import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Set

from ..llm import ChatLLMFactory
from ..tools_registry import registry

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """
    A node in the solution tree representing a single iteration.
    Stores code, execution results, and evaluation information.
    """

    # Node creation time
    ctime: float = field(default_factory=lambda: time.time())

    # Tree structure
    parent: Optional["Node"] = None
    children: Set["Node"] = field(default_factory=set)

    # Node position in tree
    time_step: int = None  # Corresponds to the global time step when created
    depth: int = 0  # Depth in the tree (root=0, increases with each level)

    # Solution stage
    stage: Literal["root", "debug", "evolve"] = "root"

    # MCTS statistics
    visits: int = 0
    validated_visits: int = 0  # Number of successful runs with validation scores
    failure_visits: int = 0  # Number of failed runs
    unvalidated_visits: int = 0  # Number of successful runs without validation scores
    validated_reward: float = 0.0  # Total reward from validated runs
    # total_reward: float = 0.0  # Replaced by separate reward tracking

    # Node state tracking
    is_successful: bool = False  # Did the execution succeed?
    is_debug_successful: bool = False  # Did the debug in the subtree succeed?
    is_terminal: bool = False  # Should this node not be expanded further?
    debug_attempts: int = 0  # Number of debug attempts on this node

    # Solution artifacts
    python_code: str = ""
    bash_script: str = ""
    tool_used: str = ""  # The primary tool used for this solution
    tools_available: List[str] = field(
        default_factory=list
    )  # All tools available for this solution, in priority order
    tutorial_retrieval: str = ""  # Retrieved tutorials for this node
    tutorial_prompt: str = ""  # Processed tutorial prompt for this node

    # Execution results
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    error_message: str = ""
    error_analysis: str = ""

    # Evaluation metrics
    validation_score: Optional[float] = None

    # Locking for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)
    expected_child_count: int = 0

    @property
    def id(
        self,
    ):
        return self.time_step

    def __post_init__(self):
        """
        Initialize a node, adding it to parent's children if parent exists.
        Set depth based on parent's depth.
        """
        if self.parent is not None:
            self.parent.add_child(self)
            self.depth = self.parent.depth + 1

    def add_child(self, child: "Node") -> None:
        """
        Add a child node to this node.
        """
        logger.detail(f"Node {child.id} is added to children of Node {self.id}.")
        self.children.add(child)

    def remove_child(self, child: "Node") -> None:
        """
        Remove a child node of this node.
        """
        logger.detail(f"Node {child.id} is removed from children of Node {self.id}.")
        self.children.remove(child)

    @property
    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf node (has no children).
        """
        return len(self.children) == 0

    @property
    def num_children(self) -> int:
        """
        Get the number of child nodes.
        """
        return len(self.children)

    @property
    def prev_tutorial_prompt(self) -> str:
        if self.parent and self.parent.tutorial_prompt:
            return self.parent.tutorial_prompt

    def update(self, reward: float, is_validated: bool = False, is_failure: bool = False) -> None:
        """
        Update the node's statistics with a new reward.

        Args:
            reward: The raw validation score (for validated runs) or None
            is_validated: Whether this reward comes from a validated run
            is_failure: Whether this run was a failure
        """
        with self._lock:
            self.visits += 1

            if is_failure:
                self.failure_visits += 1
            elif is_validated and reward is not None:
                # For validated runs, store the raw validation score
                self.validated_visits += 1
                self.validated_reward += reward  # Sum up the raw scores, will be normalized in UCT
            else:
                # For successful runs without validation
                self.unvalidated_visits += 1

    def uct_value(
        self,
        exploration_constant: float = 1.414,
        best_score: Optional[float] = None,
        worst_score: Optional[float] = None,
        failure_offset: float = 0,
        failure_penalty_weight: float = 0.5,
    ) -> float:
        """
        Calculate the UCT (Upper Confidence Bound for Trees) value of the node.

        Args:
            exploration_constant: The constant that controls exploration vs exploitation
            best_score: The best validation score seen so far (for scaling)
            worst_score: The worst validation score seen so far (for scaling)

        Returns:
            The UCT value
        """
        # For unvisited nodes, return infinity to ensure they are visited
        if self.visits == 0:
            return float("inf")

        # Get parent visits for UCT calculation
        if self.parent:
            parent_visits = max(1, self.parent.visits)
        else:
            parent_visits = 1

        # Calculate exploitation term based on node stats
        self.normalized_failure_visit = max(0, self.failure_visits - failure_offset)
        self.failure_penalty = -failure_penalty_weight * self.normalized_failure_visit / self.visits

        # Calculate the validated rewards part
        if self.validated_visits > 0:
            if best_score is not None and worst_score is not None and best_score > worst_score:
                # Normalize the validated_reward using best and worst scores
                # First get the average raw score
                self.avg_raw_score = self.validated_reward / self.validated_visits
                # Then normalize it between 0 and 1
                self.normalized_score = (self.avg_raw_score - worst_score) / (best_score - worst_score)
                self.validated_weight = self.validated_visits / self.visits
                self.validated_contribution = self.validated_weight * self.normalized_score
            else:
                # If can't normalize
                self.validated_contribution = 1.0
        else:
            self.validated_contribution = 0.0

        # Unvalidated contribution (nodes that succeeded but have no score) use a score of 0. and thus can be ignored

        # Total exploitation is the weighted sum of all components
        self.exploitation = self.validated_contribution + self.failure_penalty

        # Calculate exploration term
        self.exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)

        return self.exploitation + self.exploration

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class NodeManager:
    """
    Manages a tree of nodes representing different iterations of solution development.
    Uses Monte Carlo Tree Search (MCTS) to explore the solution space more effectively.
    """

    def __init__(
        self,
        input_data_folder: str,
        output_folder: str,
        config: Any,
        initial_user_input: str,
        enable_per_iteration_instruction: bool,
    ):
        """
        Initialize the NodeManager with required paths and configuration.

        Args:
            input_data_folder: Path to input data directory
            output_folder: Path to output directory
            config: Configuration object
            initial_user_input: Initial user instruction
            enable_per_iteration_instruction: If asking for per iteration user input
        """
        # Store required paths
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder

        # Validate paths
        for path, name in [(input_data_folder, "input_data_folder")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.config = config
        self.enable_per_iteration_instruction = enable_per_iteration_instruction
        self.initial_user_input = initial_user_input

        # Track time_step
        self.time_step = -1
        # Create root node
        self.root_node = Node(stage="root", time_step=self.time_step, depth=0)
        self.current_node = self.root_node

        # Track best nodes and metrics
        self._best_node = None
        self._best_validation_score = None
        self._worst_validation_score = None
        self.last_successful_node = None

        # Key node tracking
        self.best_step = -1
        self.last_successful_step = -1

        # MCTS parameters
        self.exploration_constant = self.config.exploration_constant
        self.max_debug_depth = self.config.max_debug_depth
        self.failure_offset = self.config.failure_offset
        self.failure_penalty_weight = self.config.failure_penalty_weight

        # Tracking for thread safety
        self._node_lock = threading.Lock()
        self.search_start_time = time.time()

        # User inputs storage
        self.user_inputs = []

        # Error analysis storage
        self._all_error_analyses = []

        # Tool tracking
        self.used_tools = set()

        # Target prompt instance for meta-prompting
        self.target_prompt_instance = None

        # Initialize the agent components
        self._init_agents()

    def _init_agents(self):
        """Initialize all required agents."""
        from ..agents import (
            CoderAgent,
            DataPerceptionAgent,
            DescriptionFileRetrieverAgent,
            ErrorAnalyzerAgent,
            ExecuterAgent,
            MetaPromptingAgent,
            RerankerAgent,
            RetrieverAgent,
            TaskDescriptorAgent,
            ToolSelectorAgent,
        )

        # Data perception agent
        self.dp_agent = DataPerceptionAgent(
            config=self.config,
            manager=self,
            input_data_folder=self.input_data_folder,
            reader_llm_config=self.config.reader,
            reader_prompt_template=None,
        )

        # Description file retriever agent
        self.dfr_agent = DescriptionFileRetrieverAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.description_file_retriever,
            prompt_template=None,
        )

        # Task descriptor agent
        self.td_agent = TaskDescriptorAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.task_descriptor,
            prompt_template=None,
        )

        # Tool selector agent
        self.ts_agent = ToolSelectorAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.tool_selector,
            prompt_template=None,
        )

        # Initialize meta-prompting
        self.enable_meta_prompting = self.config.enable_meta_prompting
        self.meta_prompting_agent = MetaPromptingAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.meta_prompting,
        )

        # Error analyzer
        self.error_analyzer = ErrorAnalyzerAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.error_analyzer,
            prompt_template=None,
        )

        # Retriever
        self.retriever = RetrieverAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.retriever,
            prompt_template=None,
        )

        # Reranker
        self.reranker = RerankerAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.reranker,
            prompt_template=None,
        )

        # Python coder
        self.python_coder = CoderAgent(
            config=self.config,
            manager=self,
            language="python",
            coding_mode="coder",
            llm_config=self.config.python_coder,
            prompt_template=None,
        )

        # Bash coder
        self.bash_coder = CoderAgent(
            config=self.config,
            manager=self,
            language="bash",
            coding_mode="coder",
            llm_config=self.config.bash_coder,
            prompt_template=None,
        )

        # Executer
        self.executer = ExecuterAgent(
            config=self.config,
            manager=self,
            language="bash",
            timeout=self.config.per_execution_timeout,
            executer_llm_config=self.config.executer,
            executer_prompt_template=None,
        )

    def initialize(self):
        """Initialize the manager."""
        self.data_prompt = self.dp_agent()
        self.description_files = self.dfr_agent()
        self.task_description = self.td_agent()

        # Use tool selector to get prioritized list of tools
        self.available_tools = self.ts_agent()

    def get_iteration_folder(self, node: Node) -> str:
        """
        Get the folder for storing iteration artifacts.

        Args:
            node: The node to get the folder for

        Returns:
            Path to the iteration folder
        """
        if node.id < 0:
            iter_folder = os.path.join(self.output_folder, "node_init")
        else:
            iter_folder = os.path.join(self.output_folder, f"node_{node.id}")
        os.makedirs(iter_folder, exist_ok=True)
        return iter_folder

    def get_per_iteration_output_folder(self, node: Node) -> str:
        """
        Get the folder for storing iteration output artifacts.

        Args:
            node: The node to get the output folder for

        Returns:
            Path to the iteration output folder
        """
        iter_output_folder = os.path.join(self.get_iteration_folder(node), "output")
        os.makedirs(iter_output_folder, exist_ok=True)
        return iter_output_folder

    def save_and_log_states(self, content, save_name, per_iteration=False, add_uuid=False, node=None):
        """
        Save states to a file and log them.

        Args:
            content: Content to save
            save_name: Name for the saved file
            per_iteration: Whether this is for a specific iteration (backward compatibility)
            add_uuid: Whether to add a UUID to the filename
            node: Node to associate with the saved content (required if per_iteration is False)
        """
        if add_uuid:
            # Split filename and extension
            name, ext = os.path.splitext(save_name)
            # Generate 4-digit UUID (using first 4 characters of hex)
            uuid_suffix = str(uuid.uuid4()).replace("-", "")[:4]
            save_name = f"{name}_{uuid_suffix}{ext}"

        # Determine the save directory
        if per_iteration and self.current_node:
            states_dir = os.path.join(self.get_iteration_folder(self.current_node), "states")
        elif node:
            states_dir = os.path.join(self.get_iteration_folder(node), "states")
        else:
            states_dir = os.path.join(self.output_folder, "states")

        os.makedirs(states_dir, exist_ok=True)
        output_file = os.path.join(states_dir, save_name)

        logger.info(f"Saving {output_file}...")
        with open(output_file, "w") as file:
            if content is not None:
                if isinstance(content, list):
                    # Join list elements with newlines
                    file.write("\n".join(str(item) for item in content))
                else:
                    # Handle as string (original behavior)
                    file.write(content)
            else:
                file.write("<None>")

    def log_agent_start(self, message: str):
        """Log agent start message."""
        logger.info(message)

    def log_agent_end(self, message: str):
        """Log agent end message."""
        logger.info(message)

    def select_node(self) -> Node:
        """
        Select a node for expansion using UCT selection.

        Returns:
            The selected node
        """
        node = self.root_node

        # Traverse the tree until we find a node to expand
        while node is not None and not node.is_leaf:
            # If the node is not fully expanded, return it
            if not self._is_fully_expanded(node):
                return node

            # Otherwise, select the best child according to UCT
            node = self._uct_select(node)

        return node

    def _is_fully_expanded(self, node: Node) -> bool:
        """
        Check if a node is fully expanded.

        Args:
            node: The node to check

        Returns:
            True if the node is fully expanded, False otherwise
        """
        # Root node
        if node.stage == "root":
            return node.num_children >= self.config.initial_root_children or self._get_unused_tool() is None

        # For debug nodes, stop expanding after getting a successful node
        if node.stage == "debug":
            # TODO: better debugging workflow?
            if node.is_debug_successful:
                return True
            return node.num_children >= self.config.max_debug_children

        # For evolve nodes
        if node.stage == "evolve":
            return node.num_children >= self.config.max_evolve_children

        return False

    def _uct_select(self, node: Node) -> Node:
        """
        Select the best child node according to UCT, excluding terminal nodes.

        Args:
            node: The parent node

        Returns:
            The selected child node
        """
        non_terminal_children = [child for child in node.children if not child.is_terminal]
        if not non_terminal_children:
            # Fallback case - this shouldn't happen if backpropagation is working correctly
            assert (
                node.is_terminal
            ), f"All children of node {node.id} are terminal but node itself is not marked terminal"
            logger.info("All nodes are terminal. Run complete.")
            return None

        # Pass the best and worst validation scores for proper scaling
        # If current node is root, adjust exploration constant based on tool index
        if node == self.root_node:
            # Get each child's tool index in the available tools list
            def get_child_uct(child):
                # Tools earlier in the list get higher exploration constants
                tool_index = self.available_tools.index(child.tool_used)
                # Scale exploration constant - earlier tools get higher values
                tool_specific_exploration = self.exploration_constant * max(0.25, 1.0 - 0.25 * tool_index)
                # Use config for failure offset
                uct_value = child.uct_value(
                    tool_specific_exploration,
                    self._best_validation_score,
                    self._worst_validation_score,
                    failure_offset=self.failure_offset,
                    failure_penalty_weight=self.failure_penalty_weight,
                )
                logger.detail(f"UCT Value is {uct_value} for Node {child.id}")
                return uct_value

            return max(non_terminal_children, key=get_child_uct)
        else:
            # For non-root nodes, use the standard exploration constant
            def get_child_uct(child):
                uct_value = self.compute_uct_value(child)
                logger.detail(f"UCT Value is {uct_value} for Node {child.id}")
                return uct_value

        return max(non_terminal_children, key=get_child_uct)

    def expand(self) -> Node:
        """
        Expand the current node by creating a child node.

        Returns:
            The newly created child node
        """
        if self.current_node.stage == "root":
            return self._create_evolve_node()
        elif self.current_node.is_successful:
            return self._create_evolve_node()
        else:
            return self._create_debug_node()

    def _get_unused_tool(self) -> Optional[str]:
        """
        Get a tool that has not been used yet in the tree.

        Returns:
            An unused tool, or None if all tools have been used
        """
        unused_tools = [tool for tool in self.available_tools if tool not in self.used_tools]
        if unused_tools:
            # return random.choice(unused_tools)
            return unused_tools[0]  # TODO: enable random selection of available tools
        return None

    def _create_debug_node(
        self,
    ) -> Node:
        """
        Create a debug node to fix issues in a failed node.

        Returns:
            The newly created debug node
        """
        # Increment global time step for this new node
        self.time_step += 1

        # Create a new node
        self.current_node = Node(
            parent=self.current_node,
            stage="debug",
            # Use the same tool as the parent for debugging
            tool_used=self.current_node.tool_used,
            tools_available=self.available_tools,
            time_step=self.time_step,
            debug_attempts=self.current_node.debug_attempts + 1,
        )

        # Check if we've exceeded the maximum debug attempts for this node
        if self.current_node.debug_attempts >= self.max_debug_depth:
            logger.warning(
                f"Node {self.current_node.id} has reached the maximum debug depth ({self.max_debug_depth}). Marking as terminal."
            )
            self.mark_node_terminal(self.current_node)

        # Generate code for the node
        self._generate_code()

    def _create_evolve_node(
        self,
    ) -> Node:
        """
        Create an evolve node to improve a successful node.

        Returns:
            The newly created evolve node
        """
        # Increment global time step for this new node
        self.time_step += 1

        # Check if there's an unused tool to try
        unused_tool = self._get_unused_tool()
        if unused_tool:
            # If there's an unused tool, create a node from the root with that tool
            logger.info(f"Found unused tool {unused_tool}, creating evolve node from root using this tool")
            parent = self.root_node
            tool_used = unused_tool
        else:
            # Otherwise evolve from the parent node
            logger.info(f"Creating evolve node from Node {self.current_node.id} using {self.current_node.tool_used}.")
            parent = self.current_node
            tool_used = self.current_node.tool_used

        self.current_node = Node(
            parent=parent,
            stage="evolve",
            tool_used=tool_used,
            tools_available=self.available_tools,
            time_step=self.time_step,
        )

        # Generate code for the node
        self._generate_code()

    def _update_tutorials(self=None):
        """
        Retrieve and update tutorials for the current selected tool.

        Args:
            node: Node to associate the tutorials with (optional)
        """
        # Retrieve tutorials
        self.current_node.tutorial_retrieval = self.retriever()

        # Rerank the retrieved tutorials
        self.current_node.tutorial_prompt = self.reranker()

        # Save to node's folder
        self.save_and_log_states(
            content=self.current_node.tutorial_retrieval,
            save_name="tutorial_retrievals.txt",
            node=self.current_node,
            add_uuid=False,
        )
        self.save_and_log_states(
            content=self.current_node.tutorial_prompt,
            save_name="tutorial_prompt.txt",
            node=self.current_node,
            add_uuid=False,
        )

    def _generate_code(self):
        """
        Generate Python and Bash code for the current node after the tool to use is specified.
        """
        logger.debug(f"Starting code generation for Node {self.current_node.id}")

        # Mark this tool as used
        self.used_tools.add(self.current_node.tool_used)
        logger.debug(f"  Tool being used: {self.current_node.tool_used}")

        # Always get user input for this step (handles both initial and per-iteration instructions)
        logger.debug(f"  Getting user input for step {self.time_step}")
        self._get_user_input_for_step()

        # Get the tool-specific prompt for the node's selected tool
        from ..tools_registry import registry

        logger.debug("  Retrieving tool info from registry")
        tool_info = registry.get_tool(self.current_node.tool_used)
        if not tool_info:
            print(self.current_node.state)
            raise ValueError(f"Tool {self.current_node.tool_used} not found in registry")

        # Get tool-specific prompt
        self.tool_prompt = tool_info.get("prompt_template", "")
        if isinstance(self.tool_prompt, list):
            self.tool_prompt = "\n".join(self.tool_prompt)

        # Get tutorials specific to this node
        logger.debug("  Starting tutorial retrieval and reranking (this may take time)...")
        self._update_tutorials()
        logger.debug("  Finished tutorial retrieval and reranking")

        # Generate Python code
        logger.debug("  Calling Python coder agent...")
        self.current_node.python_code = self.python_coder()
        logger.debug("  Finished Python code generation")

        # Write the Python code to a file
        python_file_path = os.path.join(self.get_iteration_folder(self.current_node), "generated_code.py")
        logger.debug(f"  Writing Python code to: {python_file_path}")
        with open(python_file_path, "w") as file:
            file.write(self.current_node.python_code)

        # Generate Bash script
        logger.debug("  Calling Bash coder agent...")
        self.current_node.bash_script = self.bash_coder()
        logger.debug("  Finished Bash script generation")

        # Write the Bash script to a file
        bash_file_path = os.path.join(self.get_iteration_folder(self.current_node), "execution_script.sh")
        logger.debug(f"  Writing Bash script to: {bash_file_path}")
        with open(bash_file_path, "w") as file:
            file.write(self.current_node.bash_script)

        logger.debug(f"Completed code generation for Node {self.current_node.id}")

    def _get_user_input_for_step(self):
        """Get user input for the current step.

        - For the first code generation (time_step == 0), always use initial_user_input
        - For subsequent iterations, only prompt for additional input if enable_per_iteration_instruction is True
        """
        if self.time_step == 0:
            # First iteration: always use the initial user input from CLI
            user_input = self.initial_user_input or ""
        else:
            # Subsequent iterations: only prompt if per-iteration instruction is enabled
            if self.enable_per_iteration_instruction:
                logger.info(f"Previous iteration info is stored in: {self.get_iteration_folder(self.current_node)}")
                user_input = self.initial_user_input or ""
                user_input += "\n" + input(
                    f"Enter your inputs for current node (step {self.time_step}) (press Enter to skip): "
                )
            else:
                # Reuse the initial user input for all iterations
                user_input = self.initial_user_input or ""

        self.user_inputs.append(user_input)

    def simulate(self) -> tuple:
        """
        Simulate execution of current node and evaluate the result.

        Returns:
            Tuple containing: (validation_score, is_validated, is_failure)
                validation_score: The raw validation score (or None if not available)
                is_validated: True if this run has a validation score
                is_failure: True if this run failed
        """
        # Execute the code
        planner_decision, error_summary, validation_score, planner_prompt, stderr, stdout = self.executer(
            code_to_execute=self.current_node.bash_script,
            code_to_analyze=self.current_node.python_code,
            execution_task=self.task_description,
            execution_data=self.data_prompt,
        )

        # Store execution results
        self.current_node.stdout = stdout
        self.current_node.stderr = stderr

        # Save execution outputs
        self.save_and_log_states(stderr, "stderr", node=self.current_node, add_uuid=False)
        self.save_and_log_states(stdout, "stdout", node=self.current_node, add_uuid=False)

        # Update validation score
        self.current_node.validation_score = validation_score

        # Track the best and worst validation scores for scaling in UCT calculation
        if validation_score is not None:
            # Update best validation score
            if self._best_node is None or validation_score > self._best_validation_score:
                self._best_node = self.current_node
                self._best_validation_score = validation_score
                self.best_step = self.time_step

            # Track worst validation score (initialize if not set yet)
            if not hasattr(self, "_worst_validation_score") or self._worst_validation_score is None:
                self._worst_validation_score = validation_score
            else:
                self._worst_validation_score = min(self._worst_validation_score, validation_score)

        # Determine if the execution was successful
        if planner_decision == "SUCCESS":
            self.current_node.is_successful = True
            self.last_successful_node = self.current_node
            self.last_successful_step = self.time_step
            self.current_node.error_message = ""

            # If this is a debug node, find the origin of the debug chain
            if self.current_node.stage == "debug":
                # Find the original node that started this debugging chain
                debug_origin = self._find_debug_origin(self.current_node)

                # Add this successful node as a sibling to the original buggy node
                self.current_node.parent.remove_child(self.current_node)
                self.current_node.parent = debug_origin.parent
                debug_origin.parent.add_child(self.current_node)

                self.mark_node_terminal(debug_origin)

                logger.info(
                    f"Replaced debug origin node {debug_origin.id} with successful debug node {self.current_node.id}"
                )

            # Return the raw validation score (for tracking), is_validated flag, and is_failure flag
            return (validation_score, validation_score is not None, False)
        else:
            self.current_node.is_successful = False
            self.current_node.error_message = f"stderr: {stderr}\n\n" if stderr else ""
            self.current_node.error_message += f"Error summary: {error_summary}"

            # Get error analysis
            self.current_node.error_analysis = self.error_analyzer()

            self._all_error_analyses.append(self.current_node.error_analysis)

            # If this is a debug node and it failed, check parent's debug attempts
            if self.current_node.stage == "debug" and self.current_node.parent:
                self.current_node.parent.debug_attempts += 1
                logger.warning(
                    f"Debug attempt failed. Debug attempts on parent node {self.current_node.parent.id}: {self.current_node.parent.debug_attempts}/{self.max_debug_depth}"
                )

                # If parent has reached max debug attempts, mark it as terminal
                if self.current_node.parent.debug_attempts >= self.max_debug_depth:
                    logger.warning(
                        f"Parent node {self.current_node.parent.id} has reached the maximum debug depth. Marking as terminal."
                    )
                    self.mark_node_terminal(self.current_node.parent)

            # For failures, we return None score, not validated, and is_failure=True
            return (None, False, True)

    def backpropagate(self, simulation_result):
        """
        Backpropagate the reward up the tree and update terminal status.

        Args:
            simulation_result: Tuple of (validation_score, is_validated, is_failure)
        """
        # Extract simulation results
        validation_score, is_validated, is_failure = simulation_result

        node = self.current_node
        while node is not None:
            node.update(validation_score, is_validated, is_failure)
            node = node.parent

    def step(self):
        """
        Perform one step of the Monte Carlo Tree Search.

        Returns:
            True if a successful node was found, False otherwise
        """
        # Selection: select a node to expand
        self.current_node = self.select_node()
        if self.current_node is None:
            return None

        # Expansion: create a new child node
        # Note: time_step is now incremented in the creation methods
        self.expand()

        # Simulation: execute the code and get results
        simulation_result = self.simulate()

        # Backpropagation: update node statistics
        self.backpropagate(simulation_result)

        # Generate a visualization of the node tree after each iteration
        from .node_visualizer import visualize_tree_only

        visualize_tree_only(self)

        return self.current_node.is_successful

    def mark_node_terminal(self, node):
        """
        Mark a node and all its descendants as terminal.
        Then check if any ancestors should be marked terminal.

        Args:
            node: The node to mark as terminal
        """
        # Mark the node itself and all descendants as terminal
        self._mark_subtree_terminal(node)

        # Check if any ancestors should be marked terminal
        self._check_ancestors_terminal(node.parent)

    def _mark_subtree_terminal(self, node):
        """
        Recursively mark a node and all its descendants as terminal.

        Args:
            node: The node to mark as terminal
        """
        if node.is_terminal:
            return

        node.is_terminal = True
        logger.info(f"Marking node {node.id} as terminal")

        # Recursively mark all children
        for child in node.children:
            self._mark_subtree_terminal(child)

    def _check_ancestors_terminal(self, node):
        """
        Recursively check if ancestors should be marked as terminal.
        An ancestor is terminal if fully expanded and all children are terminal.

        Args:
            node: The ancestor node to check
        """
        if node is None:
            return

        if self._is_fully_expanded(node) and all(child.is_terminal for child in node.children):
            node.is_terminal = True
            logger.info(f"Marking ancestor node {node.id} as terminal (all children terminal)")

            # Continue checking up the tree
            self._check_ancestors_terminal(node.parent)

    def _get_all_nodes(self) -> List[Node]:
        """
        Get all nodes in the tree.

        Returns:
            List of all nodes
        """
        all_nodes = []

        def _collect_nodes(node):
            all_nodes.append(node)
            for child in node.children:
                _collect_nodes(child)

        _collect_nodes(self.root_node)
        return all_nodes

    def create_best_run_copy(self):
        """Create a 'best_run' folder that symlinks to the best node folder."""
        # Determine which node to link
        target_node = None
        link_reason = ""

        if self._best_node:
            target_node = self._best_node
            link_reason = f"best validation score ({self._best_validation_score:.4f})"
        elif self.last_successful_node:
            target_node = self.last_successful_node
            link_reason = "last successful execution"
        else:
            logger.warning("No best node or successful node found. Cannot create best_run link.")
            return

        # Create paths
        source_folder = self.get_iteration_folder(target_node)
        best_run_folder = os.path.join(self.output_folder, "best_run")

        # Verify source folder exists
        if not os.path.exists(source_folder):
            logger.warning(f"Source folder does not exist: {source_folder}")
            return

        # Check if source folder has an 'output' subdirectory
        source_output_folder = os.path.join(source_folder, "output")
        if not os.path.exists(source_output_folder):
            logger.warning(f"Source output folder does not exist: {source_output_folder}")
            return

        # Handle existing best_run folder/link
        old_best_folder = None
        if os.path.exists(best_run_folder) or os.path.islink(best_run_folder):
            try:
                if os.path.islink(best_run_folder):
                    # Save the old target for potential cleanup
                    logger.debug(f"Reading existing best_run symlink target: {best_run_folder}")
                    old_link_target = os.readlink(best_run_folder)
                    old_best_folder = os.path.abspath(os.path.join(os.path.dirname(best_run_folder), old_link_target))
                    logger.debug(
                        f"Unlinking existing best_run symlink: {best_run_folder} (pointed to {old_best_folder})"
                    )
                    os.unlink(best_run_folder)
                    logger.info("Removed existing best_run symlink")
                else:
                    import shutil

                    logger.debug(f"Removing existing best_run folder (not a symlink): {best_run_folder}")
                    shutil.rmtree(best_run_folder)
                    logger.info("Removed existing best_run folder")
            except Exception as e:
                logger.error(f"Failed to remove existing best_run folder/link: {e}")
                return

        try:
            # Log completion marker
            logger.brief(
                f"Task completed successfully! Best node: {target_node.id} with validation score {target_node.validation_score}"
            )

            # Copy all files from source_output_folder to self.output_folder
            import shutil

            logger.debug(
                f"Starting copy of output folder contents from {source_output_folder} to {self.output_folder}"
            )
            for item in os.listdir(source_output_folder):
                source_item = os.path.join(source_output_folder, item)
                dest_item = os.path.join(self.output_folder, item)

                logger.debug(f"Copying item: {item}")
                if os.path.isfile(source_item):
                    logger.debug(f"  Copying file: {source_item} -> {dest_item}")
                    shutil.copy2(source_item, dest_item)
                elif os.path.isdir(source_item):
                    logger.debug(f"  Copying directory: {source_item} -> {dest_item}")
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
            logger.debug("Finished copying output folder contents")

            # Create symbolic link to the source folder instead of copying
            logger.debug(f"About to create symlink: {best_run_folder} -> {source_folder}")
            logger.info("Creating best_run symlink to best solution folder (instant operation, saves disk space)")
            os.symlink(source_folder, best_run_folder, target_is_directory=True)
            logger.debug("Successfully created best_run symlink")

            logger.info(f"Created best_run symlink (linked to node {target_node.id} - {link_reason})")

            # Save summary information in the target node folder
            summary_content = [
                "Best Run Summary",
                "================",
                f"Linked to: node_{target_node.id}",
                f"Reason: {link_reason}",
                f"Tool used: {target_node.tool_used}",
                f"Symlink created at: {os.path.basename(best_run_folder)}",
                "",
                self.get_validation_score_summary(),
                "",
                "Tool Usage Summary:",
                "==================",
                f"Available tools: {', '.join(self.available_tools)}",
                f"Tools used: {', '.join(self.used_tools)}",
                f"Tools not used: {', '.join(set(self.available_tools) - self.used_tools)}",
            ]

            # Save summary in both the main output folder and the target node folder
            summary_text = "\n".join(summary_content)

            self.save_and_log_states(
                content=summary_text, save_name="best_run_summary.txt", node=target_node, add_uuid=False
            )

            # Clean up old best folder if cleanup is enabled and it's not the same as the new one
            if old_best_folder and self.config.remove_current_iteration_folder:
                source_folder_abs = os.path.abspath(source_folder)
                logger.debug(f"Checking if old best folder should be removed: {old_best_folder}")
                logger.debug(f"  New best folder: {source_folder_abs}")
                logger.debug(f"  Are they different? {old_best_folder != source_folder_abs}")
                logger.debug(f"  Does old folder exist? {os.path.exists(old_best_folder)}")
                if old_best_folder != source_folder_abs and os.path.exists(old_best_folder):
                    try:
                        import shutil

                        logger.debug(f"About to remove old best folder: {old_best_folder}")
                        shutil.rmtree(old_best_folder)
                        logger.info(
                            f"Removed old best node folder {old_best_folder} (superseded by new best node {target_node.id})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to remove old best folder {old_best_folder}: {e}")

        except Exception as e:
            logger.error(f"Failed to create symlink: {e}")

    def remove_current_iteration_folder(self):
        """Remove the current iteration folder to save disk space.

        IMPORTANT: This should NOT be called if the current node is linked by best_run symlink,
        as it would break the symlink. Only call this for intermediate nodes that are not the best.
        """
        if not self.current_node:
            logger.warning("Current node is None.")
            return

        source_folder = self.get_iteration_folder(self.current_node)
        best_run_folder = os.path.join(self.output_folder, "best_run")

        logger.debug(f"Checking if iteration folder can be removed: {source_folder}")
        logger.debug(f"  Current node: {self.current_node.id}")
        logger.debug(f"  Best run folder: {best_run_folder}")

        # Check if best_run symlink exists and points to the current folder
        if os.path.islink(best_run_folder):
            logger.debug("  best_run is a symlink, checking target...")
            link_target = os.readlink(best_run_folder)
            # Resolve to absolute paths for comparison
            link_target_abs = os.path.abspath(os.path.join(os.path.dirname(best_run_folder), link_target))
            source_folder_abs = os.path.abspath(source_folder)

            logger.debug(f"  Symlink target (absolute): {link_target_abs}")
            logger.debug(f"  Current folder (absolute): {source_folder_abs}")
            logger.debug(f"  Are they the same? {link_target_abs == source_folder_abs}")

            if link_target_abs == source_folder_abs:
                logger.info(
                    f"Skipping removal of Node {self.current_node.id} folder - it is linked by best_run symlink."
                )
                return

        if os.path.exists(source_folder):
            import shutil

            try:
                logger.debug(f"About to remove iteration folder: {source_folder}")
                shutil.rmtree(source_folder)
                logger.info(f"Removed iteration folder of Node {self.current_node.id} to save disk space.")
            except Exception as e:
                logger.error(f"Failed to remove existing current iteration folder: {e}")
                return
        else:
            logger.debug(f"Iteration folder does not exist, nothing to remove: {source_folder}")

    def get_validation_score_summary(self) -> str:
        """
        Get a summary of all validation scores.

        Returns:
            A summary string
        """
        all_nodes = self._get_all_nodes()
        nodes_with_scores = [node for node in all_nodes if node.validation_score is not None]

        if not nodes_with_scores:
            return "No validation scores available."

        summary = ["Validation Score Summary:"]
        for node in nodes_with_scores:
            marker = " (BEST)" if node == self._best_node else ""
            summary.append(f"Node {node.id} ({node.tool_used}): {node.validation_score}{marker}")

        if self._best_node:
            summary.append(
                f"\nBest score: {self._best_validation_score:.4f} from node {self._best_node.id} using {self._best_node.tool_used}"
            )

        return "\n".join(summary)

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "retriever"):
            self.retriever.cleanup()

    def _find_debug_origin(self, node: Node) -> Optional[Node]:
        """
        Find the original node that started this debugging chain.

        Args:
            node: The current node in the debug chain

        Returns:
            The original node that started the debug chain
        """
        # Go up the tree until we find a non-debug node
        current = node
        while current.parent and current.parent.stage == "debug":
            current = current.parent

        debug_origin = current.parent
        assert not debug_origin.is_successful

        return debug_origin

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    def visualize_results(self, output_path: Optional[str] = None) -> str:
        """
        Generate a PDF visualization of the node structure.

        Args:
            output_path: Path to save the PDF. If not provided, it will be saved to
                        the output folder.

        Returns:
            The path to the generated PDF file
        """
        from .node_visualizer import visualize_results

        return visualize_results(self, output_path)

    def report_token_usage(self):
        token_usage_path = os.path.join(self.output_folder, "token_usage.json")
        usage = ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)
        total = usage["total"]
        logger.brief(
            f"Total tokens — input: {total['total_input_tokens']}, "
            f"output: {total['total_output_tokens']}, "
            f"sum: {total['total_tokens']}"
        )

        logger.info(f"Full token usage detail:\n{usage}")

    def compute_uct_value(self, node):
        return node.uct_value(
            self.exploration_constant,
            self._best_validation_score,
            self._worst_validation_score,
            failure_offset=self.failure_offset,
            failure_penalty_weight=self.failure_penalty_weight,
        )

    # Properties to maintain compatibility with Manager API
    @property
    def user_input(self) -> str:
        """Get the user input for the current step."""
        if self.time_step < 0 or self.time_step >= len(self.user_inputs):
            return ""
        return self.user_inputs[self.time_step]

    @property
    def best_validation_score(self) -> float:
        """Get the best validation score."""
        return self._best_validation_score if self._best_validation_score is not None else 0.0

    @property
    def best_node(self) -> Node:
        """Get the best node."""
        return self._best_node

    @property
    def python_code(self) -> str:
        """Get the Python code from the current node."""
        return self.current_node.python_code if self.current_node else ""

    @property
    def python_file_path(self) -> str:
        """Get the Python file path for the current node."""
        if not self.current_node:
            return ""
        return os.path.join(self.get_iteration_folder(self.current_node), "generated_code.py")

    @property
    def previous_python_code(self) -> str:
        """Get the Python code from the previous node."""
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.python_code
        return ""

    @property
    def bash_script(self) -> str:
        """Get the Bash script from the current node."""
        return self.current_node.bash_script if self.current_node else ""

    @property
    def previous_bash_script(self) -> str:
        """Get the Bash script from the previous node."""
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.bash_script
        return ""

    @property
    def error_message(self) -> str:
        """Get the error message from the current node."""
        return self.current_node.error_message if self.current_node else ""

    @property
    def previous_error_message(self) -> str:
        """Get the error message from the previous node."""
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.error_message
        return ""

    @property
    def error_analysis(self) -> str:
        """Get the error analysis from the current node."""
        return self.current_node.error_analysis if self.current_node else ""

    @property
    def previous_error_analysis(self) -> str:
        """Get the error analysis from the previous node."""
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.error_analysis
        return ""

    @property
    def all_previous_error_analyses(self) -> str:
        """Get all error analyses from previous nodes."""
        # TODO: make this recursive, handle debugging code and successful ones differently
        return "\n\n".join(self._all_error_analyses)

        if not self.current_node:
            return ""

        analyses = []
        node = self.current_node
        while node.parent:
            node = node.parent
            if node.error_analysis:
                analyses.append(node.error_analysis)

        return "\n\n".join(analyses)

    @property
    def per_iteration_output_folder(self) -> str:
        """Get the output folder for the current iteration."""
        if not self.current_node:
            return os.path.join(self.output_folder, "initialization", "output")
        return self.get_per_iteration_output_folder(self.current_node)

    @property
    def iteration_folder(self) -> str:
        """Get the folder for the current iteration."""
        if not self.current_node:
            return os.path.join(self.output_folder, "initialization")
        return self.get_iteration_folder(self.current_node)

    @property
    def tutorial_retrieval(self) -> str:
        """Get the tutorial retrieval for the current step."""
        if self.current_node:
            return self.current_node.tutorial_retrieval
        else:
            logger.warning("Invalid node while asking for tutorial_retrieval")

    @property
    def tutorial_prompt(self) -> str:
        """Get the tutorial prompt for the current step."""
        return self.current_node.tutorial_prompt if self.current_node else ""

    @property
    def previous_tutorial_prompt(self) -> str:
        """Get the tutorial prompt from the previous step."""
        return self.current_node.prev_tutorial_prompt

    @property
    def common_env_file(self) -> str:
        return registry.registry_path / "_common" / "requirements.txt"

    @property
    def selected_tool(self) -> str:
        return self.current_node.tool_used

    @property
    def selected_tool_env_file(self) -> str:
        tool_path = registry.get_tool(self.selected_tool)["path"]
        return registry.registry_path / tool_path / "requirements.txt"

    @property
    def configure_env(
        self,
    ):
        if self.selected_tool.lower() in ["machine learning", "huggingface", "fairseq"]:
            return True
        else:
            return self.config.configure_env

    @property
    def code_to_improve(
        self,
    ):
        if self.current_node.stage == "evolve":
            return self.current_node.parent.python_code
        else:
            return None

    @property
    def code_to_debug(
        self,
    ):
        if self.current_node.stage == "debug":
            return self.current_node.parent.python_code
        else:
            return None
