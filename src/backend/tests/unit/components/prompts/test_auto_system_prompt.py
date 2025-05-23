import pytest
from langflow.schema.message import Message
from langflow.components.prompts.auto_system_prompt import AutoSystemPromptComponent


@pytest.fixture
def component():
    # Simplified instantiation for testing.
    # Passing _vertex_id and _graph as None, assuming the component's build logic
    # doesn't strictly depend on a fully initialized Vertex/Graph for these tests.
    # Adjust if specific graph/vertex interactions are needed for the component's methods.
    comp = AutoSystemPromptComponent(_vertex_id="test_vertex_id", _graph=None)
    # Initialize input attributes that would normally be set by Langflow
    comp.user_goal = ""
    comp.keywords = ""
    comp.tone = "Formal"  # Default from Select
    comp.target_model_family = "Default" # Default from Select
    return comp

async def test_component_instantiation(component):
    assert component is not None
    assert component.name == "AutoSystemPrompt"

async def test_generate_prompt_all_inputs(component):
    component.user_goal = "Translate English to French"
    component.keywords = "greetings, common phrases"
    component.tone = "Formal"
    component.target_model_family = "OpenAI (GPT-family)"

    message = await component.generate_prompt()
    assert isinstance(message, Message)
    assert "Your primary goal is to: Translate English to French." in message.text
    assert "Key concepts to focus on: greetings, and common phrases." in message.text
    assert "Maintain a Formal tone in your responses." in message.text
    assert "Ensure your responses are concise and clear" in message.text
    assert component.status == "Generated prompt for: Translate English to French..."

async def test_generate_prompt_default_goal_and_keywords(component):
    component.user_goal = "" 
    component.keywords = ""
    # Tone and target_model_family will use their defaults if not set here
    component.tone = "Casual" # Explicitly set for this test case
    component.target_model_family = "Default" # Explicitly set

    message = await component.generate_prompt()
    assert isinstance(message, Message)
    assert "Your primary goal is to assist the user with their tasks." in message.text
    assert "Key concepts to focus on: all relevant concepts." in message.text
    assert "Maintain a Casual tone in your responses." in message.text
    assert "OpenAI (GPT-family)" not in message.text 
    assert "Anthropic (Claude-family)" not in message.text
    assert component.status == "Generated generic prompt."


async def test_generate_prompt_anthropic_model(component):
    component.user_goal = "Summarize technical documents"
    component.keywords = "AI, machine learning, NLP"
    component.tone = "Technical"
    component.target_model_family = "Anthropic (Claude-family)"

    message = await component.generate_prompt()
    assert isinstance(message, Message)
    assert "Your primary goal is to: Summarize technical documents." in message.text
    assert "Key concepts to focus on: AI, machine learning, and NLP." in message.text
    assert "Maintain a Technical tone in your responses." in message.text
    assert "Structure your responses clearly. You can use XML tags for structuring if appropriate." in message.text
    assert component.status == "Generated prompt for: Summarize technical documents..."

async def test_generate_prompt_no_specific_model_instruction(component):
    component.user_goal = "Write a poem"
    component.keywords = "nature, beauty"
    component.tone = "Friendly"
    component.target_model_family = "Open Source LLMs" 

    message = await component.generate_prompt()
    assert isinstance(message, Message)
    assert "OpenAI (GPT-family)" not in message.text
    assert "Anthropic (Claude-family)" not in message.text
    assert "Your primary goal is to: Write a poem." in message.text
    assert component.status == "Generated prompt for: Write a poem..."

async def test_generate_prompt_single_keyword(component):
    component.user_goal = "Explain quantum physics"
    component.keywords = "qubit"
    component.tone = "Technical"
    component.target_model_family = "Default"

    message = await component.generate_prompt()
    assert isinstance(message, Message)
    assert "Key concepts to focus on: qubit." in message.text
    assert component.status == "Generated prompt for: Explain quantum physics..."
