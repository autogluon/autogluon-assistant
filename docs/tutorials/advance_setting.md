## Advanced Settings
### Model Execution Settings
The settings above the divider line control how the model runs, while the settings below the divider line relate to the model being used (including provider, credentials, and model parameters).
### Model Execution Configuration
**Max Iterations**: The number of rounds the model will run. The program automatically stops when this limit is reached. Default is 5, adjustable as needed.
Manual Prompts Between Iterations: Choose whether to add iteration-specific prompts between iterations or not.
**Log Verbosity**: Select the level of detail for the logs you want to see. Three options are available: brief, info, and detail. Brief is recommended.
**Brief**: Contains key essential information
**Info**: Includes brief information plus detailed information such as file save locations
**Detail**: Includes info-level information plus all model training related information
### Model Configuration
You can select the LLM provider, model, and credentials to use. If using Bedrock as the provider, you can use EC2 defaults. You can also upload your own config file, which will override the provider and model name settings. Provided credentials will be validated.
### Chat Input Box
**Initial Task Submission**: When starting a task for the first time, drag the input folder into this chat input box, enter any description or requirements about the task, then press Enter or click the submit button on the right. Note: Submitting larger files may sometimes fail - you can try multiple times if needed.
**Manual Prompts**: If you selected "Manual prompts between iterations" in settings, you can input prompts here.
**Task Cancellation**: After submitting a task, if you want to cancel it, submit "cancel" in this input box.