import os
import openai
import logging

MODEL_ID = "gpt-3.5-turbo"
ASSISTANT_CONTEXT = "You are analyzing the concentration of a user, I want you to roast the person if they're not focused."


class Chat(object):
    def __init__(self, key):
        self.tot_tokens = 0
        openai.api_key = key
        self.model_id = MODEL_ID
        self.dialog = [
            {"role": "assistant",
             "content": ASSISTANT_CONTEXT}
        ]

    def __del__(self):
        logging.info(f"Total tokens used: {self.tot_tokens}")

    def chat(self, prompt=None):
        if prompt is not None:
            self.dialog.append(
                {"role": "user",
                 "content": prompt})
        #
        response = openai.ChatCompletion.create(
            model=MODEL_ID,
            messages=self.dialog
        )
        api_usage = response["usage"]
        self.tot_tokens += api_usage["total_tokens"]
        self.dialog.append(
            {"role": response.choices[0].message.role, 
            "content": response.choices[0].message.content
            })
        return response