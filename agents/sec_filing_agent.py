from agents.google_gen_ai_agent import GoogleGenAIAgent
from tools.company_tools import find_cik


class SecFilingAgent:
    """
    A specialized agent for handling SEC filing queries using Google's generative AI model.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the SecFilingAgent with the given model and a function for CIK lookup.

        :param model_name: The name of the generative AI model to be used (default is "gemini-1.5-flash").
        """
        # Define the available function(s), in this case, the CIK lookup function.
        functions = {"find_cik": find_cik}

        # Initialize the GoogleGenAIAgent with the provided model and function(s).
        self.google_gen_ai_agent = GoogleGenAIAgent(model_name=model_name, functions=functions)

    def invoke(self, user_input: str):
        """
        Process user input and generate a response using the AI agent.

        :param user_input: The user's input related to SEC filings.
        :return: The AI agent's response after processing the user input.
        """
        # Generate the prompt based on the user's input.
        prompt = self._get_prompt(user_input)

        # Invoke the AI agent with the generated prompt and limit function calls to 3.
        return self.google_gen_ai_agent.invoke(message=prompt, max_function_calling=3)

    @staticmethod
    def _get_prompt(user_input: str) -> str:
        """
        Generate a prompt for the AI model, using the user's input to create a context for SEC filing details.

        :param user_input: The input provided by the user.
        :return: A string prompt for the AI model to process.
        """
        return f"""You are an advanced SEC filing expert. From this input {user_input}, get me a company name, filing type 
        (e.g., 10-Q, 10-K, 8-K, etc.), year, and the corresponding CIK in JSON format.

        **Special Considerations:** 
        * If the year is the current year and the requested filing type is 10-K, 
          check for interim reports (e.g., 10-Q) instead.
        * Ensure you provide the best filing type that can hold the information the user wants to know 
          based on this input {user_input}.
        * If no filing is found for the current year, provide a message indicating that the filing is not yet available.
        """
