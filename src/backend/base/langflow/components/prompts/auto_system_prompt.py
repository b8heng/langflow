from langflow.custom import Component
from langflow.io import TextInput, Output, Select
from langflow.schema.message import Message

class AutoSystemPromptComponent(Component):
    display_name: str = "Auto System Prompt"
    description: str = "Generates a system prompt based on user inputs."
    icon = "Sparkles"  # Or a more suitable existing icon
    name = "AutoSystemPrompt"

    inputs = [
        TextInput(
            name="user_goal",
            display_name="User Goal",
            info="Describe the main objective for the AI.",
        ),
        TextInput(
            name="keywords",
            display_name="Keywords",
            info="Comma-separated keywords or concepts to include in the prompt.",
            advanced=True,
        ),
        Select(
            name="tone",
            display_name="Tone",
            options=["Formal", "Casual", "Technical", "Friendly", "Assertive"],
            value="Formal",
            info="Select the desired tone for the system prompt.",
            advanced=True,
        ),
        Select(
            name="target_model_family",
            display_name="Target Model Family",
            options=["Default", "OpenAI (GPT-family)", "Anthropic (Claude-family)", "Open Source LLMs"],
            value="Default",
            info="Optimize prompt structure for a specific model family (optional).",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            name="system_prompt",
            display_name="System Prompt",
            method="generate_prompt",
        ),
    ]

    async def generate_prompt(self) -> Message:
        user_goal = self.user_goal
        keywords = self.keywords
        tone = self.tone
        target_model_family = self.target_model_family

        prompt_parts = ["You are an AI assistant."]

        if user_goal:
            prompt_parts.append(f"Your primary goal is to: {user_goal}.")
        else:
            prompt_parts.append("Your primary goal is to assist the user with their tasks.")

        if keywords:
            keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
            if len(keyword_list) > 1:
                keywords_string = ", ".join(keyword_list[:-1]) + f", and {keyword_list[-1]}"
            elif keyword_list:
                keywords_string = keyword_list[0]
            else: # Handles cases like keywords = "," or keywords = " , "
                keywords_string = "all relevant concepts"
            prompt_parts.append(f"Key concepts to focus on: {keywords_string}.")
        else:
            prompt_parts.append("Key concepts to focus on: all relevant concepts.")

        prompt_parts.append(f"Maintain a {tone} tone in your responses.")

        model_specific_instructions = ""
        if target_model_family == "OpenAI (GPT-family)":
            model_specific_instructions = "Ensure your responses are concise and clear, adhering to helpfulness and harmlessness guidelines."
        elif target_model_family == "Anthropic (Claude-family)":
            model_specific_instructions = "Structure your responses clearly. You can use XML tags for structuring if appropriate."
        
        if model_specific_instructions:
            prompt_parts.append(model_specific_instructions)

        generated_prompt_string = "\n".join(prompt_parts)
        
        # Set the component's status message
        # Make sure status is a string and not a Message object
        status_message = f"Generated prompt for: {user_goal[:50]}..." if user_goal else "Generated generic prompt."
        self.status = status_message # Assign the string directly
        
        return Message(text=generated_prompt_string)
