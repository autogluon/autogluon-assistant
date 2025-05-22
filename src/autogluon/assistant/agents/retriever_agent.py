import logging
from pathlib import Path
from typing import List, Optional

from ..prompts import RetrieverPrompt
from ..tools_registry import TutorialInfo
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """
    Agent for retrieving and selecting relevant tutorials based on task context.
    
    Agent Input: Task context, data info, user prompt, error info
    Agent Output: Formatted tutorial prompt with selected relevant tutorials
    """
    
    def __init__(self, config, llm_config, prompt_template):
        super().__init__(config=config)
        self.retrieval_llm_config = llm_config
        self.retrieval_prompt_template = prompt_template
        self.retrieval_prompt = RetrieverPrompt(
            llm_config=self.retrieval_llm_config,
            template=self.retrieval_prompt_template,
        )
        
        if self.retrieval_llm_config.multi_turn:
            self.retrieval_llm = init_llm(
                llm_config=self.retrieval_llm_config,
                agent_name="retrieval",
                multi_turn=self.retrieval_llm_config.multi_turn,
            )
    
    def __call__(self, prompt_generator):
        """Select relevant tutorials and format them into a prompt."""
        # Build prompt for tutorial selection
        prompt = self.retrieval_prompt.build(prompt_generator)
        
        if not self.retrieval_llm_config.multi_turn:
            self.retrieval_llm = init_llm(
                llm_config=self.retrieval_llm_config,
                agent_name="retrieval",
                multi_turn=self.retrieval_llm_config.multi_turn,
            )
        
        response = self.retrieval_llm.assistant_chat(prompt)
        selected_tutorials = self.retrieval_prompt.parse(response)
        
        # Generate tutorial prompt using selected tutorials
        tutorial_prompt = self._generate_tutorial_prompt(selected_tutorials)
        
        return tutorial_prompt
    
    def _format_tutorial_content(
        self,
        tutorial: TutorialInfo,
        max_length: int,
    ) -> str:
        """Format a single tutorial's content with truncation if needed."""
        try:
            with open(tutorial.path, "r", encoding="utf-8") as f:
                content = f.read()

            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + "\n...(truncated)"

            formatted = f"""### {tutorial.title}
            
            {content}
            """
            return formatted

        except Exception as e:
            logger.warning(f"Error formatting tutorial {tutorial.path}: {e}")
            return ""

    def _generate_tutorial_prompt(self, selected_tutorials: List) -> str:
        """Generate formatted tutorial prompt from selected tutorials."""
        
        if not selected_tutorials:
            return ""
        
        # Get max tutorial length from config if available
        max_tutorial_length = self.config.max_tutorial_length
        
        # Format selected tutorials
        formatted_tutorials = []
        for tutorial in selected_tutorials:
            formatted = self._format_tutorial_content(tutorial, max_tutorial_length)
            if formatted:
                formatted_tutorials.append(formatted)
        
        if not formatted_tutorials:
            return ""
        
        # Save results if output folder is specified
        output_folder = self.config.output_folder
        if output_folder:
            self._save_selection_results(
                Path(output_folder), 
                selected_tutorials, 
                formatted_tutorials
            )
        
        return "\n\n".join(formatted_tutorials)
    
    def _save_selection_results(self, output_folder: Path, selected_tutorials: List, formatted_tutorials: List[str]) -> None:
        """Save selection results to output folder."""
        import json
        
        try:
            output_folder.mkdir(parents=True, exist_ok=True)
            
            selection_data = [
                {
                    "path": str(tutorial.path),
                    "title": tutorial.title,
                    "summary": tutorial.summary,
                }
                for tutorial in selected_tutorials
            ]
            
            with open(output_folder / "selected_tutorials.json", "w", encoding="utf-8") as f:
                json.dump(selection_data, f, indent=2)
            
            contents_folder = output_folder / "tutorial_contents"
            contents_folder.mkdir(exist_ok=True)
            
            for i, content in enumerate(formatted_tutorials, 1):
                with open(contents_folder / f"tutorial_{i}.md", "w", encoding="utf-8") as f:
                    f.write(content)
                    
        except Exception as e:
            logger.error(f"Error saving selection results: {e}")
