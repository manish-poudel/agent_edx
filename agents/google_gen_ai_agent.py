# Custom class to interface with Google's Generative AI model

import os
import google.generativeai as genai
from dotenv import load_dotenv
from google.protobuf.struct_pb2 import Struct


def call_function(function_call, functions):
    """
    Call the function passed by the model's response.

    :param function_call: The function call object from the model's response.
    :param functions: Dictionary mapping function names to actual functions.
    :return: The result of the called function.
    """
    function_name = function_call.name
    function_args = function_call.args
    return functions[function_name](**function_args)


class GoogleGenAIAgent:
    """
    A class to interact with Google Generative AI models and handle function calls from the model's response.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", functions=None, api_key=None):
        """
        Initialize the agent with the model and API configuration.

        :param model_name: The AI model name to be used. Defaults to "gemini-1.5-flash".
        :param functions: Dictionary of functions that the model can call. Defaults to an empty dictionary.
        :param api_key: API key for authenticating with Google. Defaults to loading from .env file.
        """

        # Set up functions that the model can call; default to an empty dictionary if not provided.
        if functions is None:
            functions = {}
        self.functions = functions

        # Load API key from environment if not passed.
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")

        # Configure Google generative AI with the provided API key.
        genai.configure(api_key=api_key)

        # Initialize counters and lists for merging responses and tracking function calls.
        self.function_calling_num: int = 0
        self.merged_parts = []
        self.merged_responses = []

        # Create the model instance using the specified model name and tools (functions).
        self.model = genai.GenerativeModel(model_name=model_name, tools=functions.values())

    def invoke(self, message, max_function_calling: int = 5):
        """
        Send a user message to the model and generate a response.

        :param message: The user input message to be processed.
        :param max_function_calling: The maximum number of times the model can call functions. Defaults to 5.
        :return: The model's response or the result of function calls.
        """
        messages = [{"role": "user", "parts": [message]}]
        return self._generate_content(messages=messages, max_function_calling=max_function_calling)

    def _generate_content(self, messages, max_function_calling: int):
        """
        Generate content based on the provided messages and handle function calls recursively.

        :param messages: List of message objects including user and model responses.
        :param max_function_calling: The maximum number of allowed function calls.
        :return: The model's response or final result after function calls.
        """
        self.function_calling_num += 1

        # Get the AI model's response for the provided messages.
        response = self.model.generate_content(messages)

        # Extract parts of the response content.
        parts = response.candidates[0].content.parts

        # If no function call in the response, return the response.
        if not parts[0].function_call:
            print(f"Content generated successfully {self.function_calling_num} time(s)")
            return response

        # If the function call limit is exceeded, return the response.
        if self.function_calling_num > max_function_calling:
            print(f"Exceeded max function calls: {self.function_calling_num} >= {max_function_calling}")
            return response

        # Extend merged parts with response parts.
        self.merged_parts.extend(parts)

        # Process each part for function calls.
        for part in response.parts:
            if fn := part.function_call:
                print(f"Invoking function {part.function_call.name}")
                result = call_function(fn, self.functions)
                print(f"Result from {part.function_call.name}: {result}")

                # Store the function result as a response part using protobuf Struct.
                s = Struct()
                s.update({"result": result})
                self.merged_responses.append(genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(name=part.function_call.name, response=s)
                ))

        # Update messages with new model parts and function call responses.
        messages += [
            {"role": "model", "parts": self.merged_parts},
            {"role": "user", "parts": self.merged_responses}
        ]

        # Recursively call _generate_content with updated messages.
        print("Generating content with updated messages...")
        return self._generate_content(messages=messages, max_function_calling=max_function_calling)
