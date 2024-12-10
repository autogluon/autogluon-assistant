from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from autogluon.assistant.constants import (
    METRICS_DESCRIPTION,
    NO_FILE_IDENTIFIED,
    NO_ID_COLUMN_IDENTIFIED,
    PROBLEM_TYPES,
)
from autogluon.assistant.prompting.utils import get_outer_columns
from autogluon.assistant.utils import is_text_file


class PromptGenerator(ABC):
    fields = None

    def __init__(self, data_description: str = ""):
        self.data_description = data_description
        self.parser = self.create_parser()

    @property
    def system_prompt(self):
        return "You are an expert assistant that parses information about data science tasks, such as data science competitions."

    @property
    def basic_intro_prompt(self):
        return "The following sections contain descriptive information about a data science task:"

    @property
    def data_description_prompt(self):
        return f"# Data Description\n{self.data_description}"

    @abstractmethod
    def generate_prompt(self) -> str:
        pass

    def get_field_parsing_prompt(self) -> str:
        return (
            f"Based on the above information, provide the correct values for the following fields strictly "
            f"in valid JSON format: {', '.join(self.fields)}.\n\n"
            "Important:\n"
            "1. Return only valid JSON. No extra explanations, text, or comments.\n"
            "2. Ensure that the output can be parsed by a JSON parser directly.\n"
            "3. Do not include any non-JSON text or formatting outside the JSON object."
            '4. An example is \{"<provided_field>": "<correct_value_for_the_field>"\}'
        )

    def generate_chat_prompt(self):
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self.generate_prompt()),
            ]
        )
        return chat_prompt

    def create_parser(self):
        response_schemas = [
            ResponseSchema(name=field, description=f"The {field} for the task") for field in self.fields
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)


class DescriptionFileNamePromptGenerator(PromptGenerator):
    fields = ["data_description_file", "evaluation_description_file"]

    def __init__(self, filenames: list):
        super().__init__()
        self.filenames = filenames

    def read_file_safely(self, filename: Path) -> Union[str, None]:
        try:
            return filename.read_text()
        except UnicodeDecodeError:
            return None

    def generate_prompt(self) -> str:
        file_content_prompts = "# Available Files And Content in The File\n\n"
        for filename in map(Path, self.filenames):
            if is_text_file(filename):
                content = self.read_file_safely(filename)
                if content is not None:
                    truncated_contents = content[:100].strip()
                    if len(content) > 100:
                        truncated_contents += "..."
                    file_content_prompts += f"File:\n\n{filename} Truncated Content:\n{truncated_contents}\n\n"
        file_content_prompts += f"Please return the full path of the file to describe the problem settings, and response with the value {NO_FILE_IDENTIFIED} if there's no such file."

        return "\n\n".join(
            [
                self.basic_intro_prompt,
                file_content_prompts,
                self.get_field_parsing_prompt(),
            ]
        )


class DataFileNamePromptGenerator(PromptGenerator):
    fields = ["train_data", "test_data", "sample_submission_data"]

    def __init__(self, data_description: str, filenames: list):
        super().__init__(data_description)
        self.filenames = filenames

    def generate_prompt(self) -> str:
        file_content_prompts = "# Available Data Files And Columns in The File\n\n"
        for filename in self.filenames:
            file_content_prompts += f"File:\n\n{filename}"

        file_content_prompts += f"Based on the data description, what are the training, test, and output data? The output file may contain keywords such as benchmark, submission, or output. Please return the full path of the data files as provided, and response with the value {NO_FILE_IDENTIFIED} if there's no such File."

        return "\n\n".join(
            [
                self.basic_intro_prompt,
                file_content_prompts,
                self.get_field_parsing_prompt(),
            ]
        )


class LabelColumnPromptGenerator(PromptGenerator):
    fields = ["label_column"]

    def __init__(self, data_description: str, column_names: list):
        super().__init__(data_description)
        self.column_names = get_outer_columns(column_names)

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                f"Based on the data description, which one of these columns is likely to be the label column:\n{', '.join(self.column_names)}",
                self.get_field_parsing_prompt(),
            ]
        )


class ProblemTypePromptGenerator(PromptGenerator):
    fields = ["problem_type"]

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                f"Based on the information provided, identify the correct problem_type to be used from among these KEYS: {', '.join(PROBLEM_TYPES)}",
                self.get_field_parsing_prompt(),
            ]
        )


class IDColumnPromptGenerator(PromptGenerator):
    fields = ["id_column"]

    def __init__(self, data_description: str, column_names: list, label_column: str):
        super().__init__(data_description)
        self.column_names = get_outer_columns(column_names)
        self.label_column = label_column

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                f"Based on the data description, which one of these columns is likely to be the Id column:\n{', '.join(self.column_names)}",
                f"If no reasonable Id column is present, for example if all the columns appear to be similarly named feature columns, "
                f"response with the value {NO_ID_COLUMN_IDENTIFIED}",
                f"ID columns can't be {self.label_column}",
                self.get_field_parsing_prompt(),
            ]
        )


class TestIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["test_id_column"]


class TrainIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["train_id_column"]


class OutputIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["output_id_column"]


class EvalMetricPromptGenerator(PromptGenerator):
    fields = ["eval_metric"]

    def __init__(self, data_description: str, metrics: str):
        super().__init__(data_description)
        self.metrics = metrics

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                f"""
Based on the information provided, identify the correct evaluation metric to be used from among these KEYS:
{', '.join(self.metrics)}
The descriptions of these metrics are:
{', '.join([METRICS_DESCRIPTION[metric] for metric in self.metrics])}
respectively.
If the exact metric is not in the list provided, then choose the metric that you think best approximates the one in the task description.
Only respond with the exact names of the metrics mentioned in KEYS. Do not respond with the metric descriptions.
""",
                self.get_field_parsing_prompt(),
            ]
        )
