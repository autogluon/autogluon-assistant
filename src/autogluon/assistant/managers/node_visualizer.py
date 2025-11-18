"""
Node visualization utility for Node Manager.

This module provides functionality to generate a PDF visualization of the node structures
created during the Monte Carlo Tree Search process in the NodeManager. The visualization
includes a tree view of nodes with their basic information and states.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .node_manager import Node, NodeManager

# For compatibility with updated node_manager.py that stores error analyses differently

logger = logging.getLogger(__name__)


class NodeTree(Flowable):
    """A Flowable for drawing the node tree structure."""

    def __init__(self, root_node: Node, node_info: Dict[int, Dict], width=700, height=500):
        Flowable.__init__(self)
        self.root_node = root_node
        self.node_info = node_info
        self.width = width
        self.height = height

    def draw(self):
        """Draw the node tree."""
        # Calculate positions for all nodes
        positions = self._calculate_positions(self.root_node)

        # Draw connections first (so they appear behind nodes)
        self._draw_connections(positions)

        # Draw nodes
        for node_id, (x, y) in positions.items():
            self._draw_node(node_id, x, y)

    def _calculate_positions(self, root: Node) -> Dict[int, Tuple[float, float]]:
        """Calculate positions for all nodes in the tree."""
        positions = {}
        max_depth = self._get_max_depth(root)

        # Get nodes by level to calculate width distribution
        nodes_by_level = [[] for _ in range(max_depth + 1)]

        def collect_by_level(node, level=0):
            if len(nodes_by_level) <= level:
                nodes_by_level.append([])
            nodes_by_level[level].append(node)
            for child in node.children:
                collect_by_level(child, level + 1)

        collect_by_level(root)

        # Level height - ensure enough vertical space between levels
        level_height = self.height / (max_depth + 2)

        # Create a mapping to track node positions
        for level, level_nodes in enumerate(nodes_by_level):
            num_nodes = len(level_nodes)
            if num_nodes == 0:
                continue

            # Use the full width with margins
            usable_width = self.width * 0.9
            margin = (self.width - usable_width) / 2

            # Calculate spacing between nodes at this level
            if num_nodes == 1:
                # If only one node at this level, center it
                x = self.width / 2
                y = self.height - (level * level_height) - (level_height / 2)
                positions[level_nodes[0].id] = (x, y)
            else:
                # Distribute nodes evenly across the usable width
                spacing = usable_width / (num_nodes - 1) if num_nodes > 1 else usable_width

                for i, node in enumerate(level_nodes):
                    x = margin + (i * spacing)
                    y = self.height - (level * level_height) - (level_height / 2)
                    positions[node.id] = (x, y)

        return positions

        return positions

    def _get_max_depth(self, root: Node) -> int:
        """Get the maximum depth of the tree."""
        if not root.children:
            return 0

        return 1 + max(self._get_max_depth(child) for child in root.children)

    def _draw_connections(self, positions: Dict[int, Tuple[float, float]]):
        """Draw connections between nodes."""
        for node_id, (x, y) in positions.items():
            node_data = self.node_info.get(node_id, {})
            parent_id = node_data.get("parent_id")

            if parent_id is not None and parent_id in positions:
                parent_x, parent_y = positions[parent_id]

                # Draw line from parent to child
                self.canv.setStrokeColor(colors.grey)
                self.canv.setLineWidth(0.5)
                self.canv.line(parent_x, parent_y, x, y)

    def _draw_node(self, node_id: int, x: float, y: float):
        """Draw a single node."""
        # Node info
        node_data = self.node_info.get(node_id, {})

        # Determine node color based on status
        if node_data.get("is_successful", False):
            fill_color = colors.lightgreen
        elif node_data.get("error_message"):
            fill_color = colors.lightcoral
        else:
            fill_color = colors.lightblue

        # Draw node circle - using much smaller radius
        radius = 12
        self.canv.setFillColor(fill_color)
        self.canv.setStrokeColor(colors.black)
        self.canv.circle(x, y, radius, stroke=1, fill=1)

        # Draw node ID - smaller font
        self.canv.setFillColor(colors.black)
        self.canv.setFont("Helvetica", 8)
        self.canv.drawCentredString(x, y - 2, str(node_id))

        # Draw validation score below the node if available
        validation_score = node_data.get("validation_score")
        if validation_score is not None:
            self.canv.setFont("Helvetica", 7)
            # Format validation score to 2 decimal places
            score_text = f"{validation_score:.2f}"
            self.canv.drawCentredString(x, y - radius - 8, score_text)

        # Standard PDF link approach

        # Add clickable area with larger detection area for easier clicking
        click_padding = 4
        # Use internal PDF links
        self.canv.linkURL(
            f"#node_{node_id}",
            (
                x - radius - click_padding,
                y - radius - click_padding,
                x + radius + click_padding,
                y + radius + click_padding,
            ),
            relative=0,
        )


class NodeVisualizer:
    """
    A utility class for generating PDF visualizations of node structures.
    """

    def __init__(self, node_manager: NodeManager):
        """
        Initialize the NodeVisualizer.

        Args:
            node_manager: The NodeManager instance
        """
        self.node_manager = node_manager
        self.styles = getSampleStyleSheet()

        # Create custom styles
        self.styles.add(ParagraphStyle("NodeTitle", parent=self.styles["Heading1"], fontSize=14, spaceAfter=10))
        self.styles.add(ParagraphStyle("NodeInfo", parent=self.styles["Normal"], fontSize=10, spaceAfter=5))
        self.styles.add(ParagraphStyle("ErrorText", parent=self.styles["Normal"], textColor=colors.red, fontSize=10))
        self.styles.add(
            ParagraphStyle(
                "NavInstruction",
                parent=self.styles["Normal"],
                fontName="Helvetica-Oblique",
                fontSize=10,
                textColor=colors.darkblue,
            )
        )

    def _create_node_info_dict(self, node: Node) -> Dict:
        """
        Create a dictionary with node information.

        Args:
            node: The node to extract information from

        Returns:
            A dictionary with node information
        """
        return {
            "id": node.id,
            "stage": node.stage,
            "time_step": node.time_step,
            "depth": node.depth,
            "is_successful": node.is_successful,
            "is_terminal": node.is_terminal,
            "tool_used": node.tool_used,
            "debug_attempts": node.debug_attempts,
            "error_message": node.error_message,
            "error_analysis": node.error_analysis,
            "validation_score": node.validation_score,
            "parent_id": node.parent.id if node.parent else None,
            "child_ids": [child.id for child in node.children],
        }

    def _get_all_nodes(self) -> List[Node]:
        """Get all nodes in the tree."""
        return self.node_manager._get_all_nodes()

    def _create_node_summary(self, node: Node) -> List[Flowable]:
        """
        Create a summary of the node.

        Args:
            node: The node to create a summary for

        Returns:
            A list of flowables for the PDF
        """
        elements = []

        # Create a bookmark/anchor for this node
        # Create anchor for internal links
        elements.append(Paragraph(f'<a name="node_{node.id}"></a>', self.styles["Normal"]))

        # Node title
        elements.append(Paragraph(f"Node {node.id} ({node.stage})", self.styles["NodeTitle"]))

        # Dynamically get all properties of the node (except special ones like parent/children which need special handling)
        node_attrs = {}
        # Extended special_attrs to exclude large text content that would break the PDF layout
        special_attrs = {
            "parent",
            "children",
            "error_message",
            "error_analysis",
            "_lock",
            "python_code",
            "bash_script",
            "stdout",
            "stderr",
            "tutorial_retrieval",
            "tutorial_prompt",
        }
        for attr_name in dir(node):
            if (not attr_name.startswith("_") or attr_name == "_lock") and attr_name not in special_attrs:
                try:
                    attr_value = getattr(node, attr_name)
                    # Skip methods
                    if not callable(attr_value):
                        node_attrs[attr_name] = attr_value
                except Exception as e:
                    node_attrs[attr_name] = f"<Error retrieving value: {str(e)}>"

        # Create a table for node properties
        property_data = [["Property", "Value"]]
        property_data.append(["uct_value", f"{self.node_manager.compute_uct_value(node):.4f}"])

        # Helper function to truncate long strings
        def truncate_string(s, max_len=500):
            if isinstance(s, str) and len(s) > max_len:
                return s[:max_len] + f" ... [truncated, {len(s)} chars total]"
            return s

        # Add all properties to the table
        for prop_name, prop_value in sorted(node_attrs.items()):
            # Format special cases
            if prop_name == "validation_score" and prop_value is not None:
                prop_value = f"{prop_value:.4f}"
            elif prop_name == "ctime":
                from datetime import datetime

                prop_value = datetime.fromtimestamp(prop_value).strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(prop_value, list) or isinstance(prop_value, set):
                # Truncate list/set representations if they're too long
                list_str = str(list(prop_value))
                prop_value = truncate_string(list_str)
            elif isinstance(prop_value, bool):
                prop_value = "✓" if prop_value else "✗"
            elif isinstance(prop_value, str):
                # Truncate any string values
                prop_value = truncate_string(prop_value)

            property_data.append([prop_name, str(prop_value)])

        # Add parent and children info
        property_data.append(["parent_id", str(node.parent.id) if node.parent else "None"])
        property_data.append(["child_ids", str([child.id for child in node.children]) if node.children else "[]"])

        # Create and style the table
        props_table = Table(property_data, colWidths=[150, 350])
        props_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (1, 0), colors.black),
                    ("ALIGN", (0, 0), (0, -1), "LEFT"),
                    ("ALIGN", (1, 0), (1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )

        elements.append(props_table)
        elements.append(Spacer(1, 10))

        # Error information if failed
        if not node.is_successful and node.error_message:
            elements.append(Paragraph("Error Message:", self.styles["Heading3"]))
            # Truncate long error messages for better PDF layout
            truncated_message = node.error_message
            if len(truncated_message) > 2000:
                truncated_message = (
                    truncated_message[:2000] + f"... [truncated, {len(node.error_message)} chars total]"
                )

            elements.append(Paragraph(truncated_message.replace("\n", "<br/>"), self.styles["ErrorText"]))

        # Error analysis if available
        if node.error_analysis:
            elements.append(Paragraph("Error Analysis:", self.styles["Heading3"]))
            # Truncate long error analyses for better PDF layout
            truncated_analysis = node.error_analysis
            if len(truncated_analysis) > 2000:
                truncated_analysis = (
                    truncated_analysis[:2000] + f"... [truncated, {len(node.error_analysis)} chars total]"
                )

            elements.append(Paragraph(truncated_analysis.replace("\n", "<br/>"), self.styles["Normal"]))

        elements.append(Spacer(1, 20))
        return elements

    def visualize_tree_only(self, output_path: Optional[str] = None) -> str:
        """
        Generate a PDF visualization of just the node tree structure (without node details).

        Args:
            output_path: Path to save the PDF. If not provided, it will be saved to
                        the node manager's output folder with a name indicating the iteration.

        Returns:
            The path to the generated PDF file
        """
        # Set default output path
        if output_path is None:
            # Use the current time step to create a unique filename
            time_step = self.node_manager.time_step
            output_path = os.path.join(self.node_manager.output_folder, f"node_tree_iteration_{time_step}.pdf")

        # Get all nodes
        all_nodes = self._get_all_nodes()

        # Create PDF document
        doc = SimpleDocTemplate(
            output_path, pagesize=landscape(letter), topMargin=20, bottomMargin=20, leftMargin=20, rightMargin=20
        )

        # Main elements for the document
        elements = []

        # Add title
        elements.append(Paragraph(f"Node Tree (Iteration {self.node_manager.time_step})", self.styles["Heading1"]))

        # Add summary statistics
        best_score_text = (
            f"Best Validation Score: {self.node_manager.best_validation_score:.4f} (Node {self.node_manager.best_step})"
            if self.node_manager.best_validation_score > 0
            else "No validation scores available"
        )

        elements.append(Paragraph(f"Total Nodes: {len(all_nodes)} | {best_score_text}", self.styles["Normal"]))
        elements.append(Spacer(1, 10))

        # Create node info dictionary for the tree visualization
        node_info = {node.id: self._create_node_info_dict(node) for node in all_nodes}

        # Add the tree visualization - sized to fit within page margins
        elements.append(NodeTree(self.node_manager.root_node, node_info, width=700, height=500))
        elements.append(Spacer(1, 20))

        # Add legend
        legend_data = [["Status", "Color"], ["Success", "Green"], ["Failure", "Red"], ["Neutral", "Blue"]]
        legend_table = Table(legend_data, colWidths=[100, 100])
        legend_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (1, 0), colors.whitesmoke),
                    ("BACKGROUND", (1, 1), (1, 1), colors.lightgreen),
                    ("BACKGROUND", (1, 2), (1, 2), colors.lightcoral),
                    ("BACKGROUND", (1, 3), (1, 3), colors.lightblue),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        elements.append(legend_table)

        # Build the document
        doc.build(elements)
        logger.info(f"Node tree visualization generated at: {output_path}")

        return output_path

    def visualize_nodes(self, output_path: Optional[str] = None) -> str:
        """
        Generate a PDF visualization of the node structure.

        Args:
            output_path: Path to save the PDF. If not provided, it will be saved to
                        the node manager's output folder.

        Returns:
            The path to the generated PDF file
        """
        # Set default output path
        if output_path is None:
            output_path = os.path.join(self.node_manager.output_folder, "node_visualization.pdf")

        # Get all nodes
        all_nodes = self._get_all_nodes()

        # Create PDF document
        doc = SimpleDocTemplate(
            output_path, pagesize=landscape(letter), topMargin=20, bottomMargin=20, leftMargin=20, rightMargin=20
        )

        # Main elements for the document
        elements = []

        # Add title with hyperlink guide
        elements.append(Paragraph("Node Structure Visualization", self.styles["Heading1"]))
        elements.append(
            Paragraph(
                "Click on any node in the tree to jump to its detailed information.", self.styles["NavInstruction"]
            )
        )

        # Add summary statistics
        best_score_text = (
            f"Best Validation Score: {self.node_manager.best_validation_score:.4f} (Node {self.node_manager.best_step})"
            if self.node_manager.best_validation_score > 0
            else "No validation scores available"
        )

        elements.append(Paragraph(f"Total Nodes: {len(all_nodes)} | {best_score_text}", self.styles["Normal"]))
        elements.append(Spacer(1, 10))

        # Create a tree visualization
        elements.append(Paragraph("Node Tree:", self.styles["Heading2"]))
        elements.append(Spacer(1, 10))

        # Create node info dictionary for the tree visualization
        node_info = {node.id: self._create_node_info_dict(node) for node in all_nodes}

        # Add the tree visualization - wider and taller to accommodate more nodes
        elements.append(NodeTree(self.node_manager.root_node, node_info, width=750, height=400))
        elements.append(Spacer(1, 20))

        # Add legend
        legend_data = [["Status", "Color"], ["Success", "Green"], ["Failure", "Red"], ["Neutral", "Blue"]]
        legend_table = Table(legend_data, colWidths=[100, 100])
        legend_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (1, 0), colors.whitesmoke),
                    ("BACKGROUND", (1, 1), (1, 1), colors.lightgreen),
                    ("BACKGROUND", (1, 2), (1, 2), colors.lightcoral),
                    ("BACKGROUND", (1, 3), (1, 3), colors.lightblue),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        elements.append(legend_table)
        elements.append(Spacer(1, 30))

        # Add individual node details
        elements.append(PageBreak())
        elements.append(Paragraph("Node Details:", self.styles["Heading2"]))
        elements.append(Spacer(1, 20))

        # Sort nodes by ID for easier navigation
        all_nodes = sorted(all_nodes, key=lambda n: n.id)

        # Add node details
        for node in all_nodes:
            elements.extend(self._create_node_summary(node))
            if node != all_nodes[-1]:  # No page break after the last node
                elements.append(PageBreak())

        # Create a table of contents
        elements.insert(5, Paragraph("Table of Contents:", self.styles["Heading2"]))
        toc_data = [["Node ID", "Stage", "Status", "Tool"]]

        # Add entry for each node to the TOC with hyperlinks
        for node in all_nodes:
            status = "✓" if node.is_successful else ("✗" if node.error_message else "")
            toc_data.append(
                [f"<a href='#node_{node.id}' color='blue'>{node.id}</a>", node.stage, status, node.tool_used or ""]
            )

        # Create the TOC table
        toc_table = Table(toc_data, colWidths=[50, 70, 50, 200])
        toc_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )

        elements.insert(6, toc_table)
        elements.insert(7, Spacer(1, 20))

        # Build the document
        doc.build(elements)
        logger.info(f"Node visualization generated at: {output_path}")

        return output_path


def visualize_tree_only(node_manager: NodeManager, output_path: Optional[str] = None) -> str:
    """
    Generate a PDF visualization of just the node tree structure (without node details).

    Args:
        node_manager: The NodeManager instance
        output_path: Path to save the PDF. If not provided, it will be saved to
                    the node manager's output folder with a name indicating the iteration.

    Returns:
        The path to the generated PDF file
    """
    visualizer = NodeVisualizer(node_manager)
    return visualizer.visualize_tree_only(output_path)


def visualize_results(node_manager: NodeManager, output_path: Optional[str] = None) -> str:
    """
    Generate a PDF visualization of the node structure.

    Args:
        node_manager: The NodeManager instance
        output_path: Path to save the PDF. If not provided, it will be saved to
                    the node manager's output folder.

    Returns:
        The path to the generated PDF file
    """
    visualizer = NodeVisualizer(node_manager)
    return visualizer.visualize_nodes(output_path)
