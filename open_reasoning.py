from typing import Optional, Dict, List
import textwrap
from openai import OpenAI
import json
import time

# Initialize OpenAI client (requires OPENAI_API_KEY in environment)
client = OpenAI(base_url="http://localhost:4000/v1",
                api_key="NONE",)

def generate(messages: List[Dict[str, str]], temp: float = 0.3, max_tokens: int = 200) -> str:
    """
    Generate text using the OpenAI API with a full conversation context.
    """
    if max_tokens is not None:
        conf = {
            "model":"gpt-4.1",
            "messages":messages,
            "temperature":temp,
            }
    else:
        conf = {
            "model":"gpt-4.1",
            "messages":messages,
            "max_tokens":max_tokens,
            "temperature":temp,
            }

    response = client.chat.completions.create(**conf)

    return response.choices[0].message.content.strip()


class ReasoningWrapper:
    """
    Wraps OpenAI model calls and implements reasoning + feedback loop.
    Returns a dict compatible with OpenAI API responses, plus
    `reasoning_history` and `feedback_reports`.
    """
    def __init__(self, model: str = "gpt-4.1-mini", reasoning_depth: int = 3):
        self.model = model
        self.reasoning_depth = reasoning_depth

    def _split_context_and_request(self, messages: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], str]:
        """
        Splits messages into conversation context and the latest user request.
        """
        latest_user_msg = None
        context_messages = []

        for msg in messages:
            if msg["role"] == "user":
                latest_user_msg = msg["content"]
            else:
                context_messages.append(msg)

        if latest_user_msg is None:
            raise ValueError("No user message found in messages")

        # Context = all messages except the final user message
        context = messages[:-1]
        return context, latest_user_msg

    def _generate_reasoning(self, context: List[Dict[str, str]], user_request: str, feedback: Optional[str] = None) -> str:
        reasoning_instructions = (
            f"[BEGIN HIDDEN REASONING]\n"
            f"Conversation context (previous turns):\n{context}\n\n"
            f"Current user request:\n{user_request}\n\n"
            "Instruction: Do only output the structured result for the four reasoning steps and not the full answer.\n\n"
            "Instruction: Keep the reasoning output as short as possible but be precise.\n\n"
            "Instruction: Provide the reason if you make assumptions.\n\n"
            "Reasoning steps:\n"
            "1. Identify the question.\n"
            "2. Break the problem into smaller parts.\n"
            "3. Solve each part.\n"
            "4. Combine the solutions.\n"
        )
        if feedback:
            reasoning_instructions += f"\n[REFINEMENT FEEDBACK]: {feedback}\n"

        reasoning_messages = context + [
            {"role": "user", "content": reasoning_instructions}
        ]

        return generate(reasoning_messages)

    def _generate_answer(self, context: List[Dict[str, str]], user_request: str, hidden_reasoning: str, visible_token_limit: int = 100) -> str:
        answer_instructions = (
            f"Conversation context:\n{context}\n\n"
            f"User request:\n{user_request}\n\n"
            f"[HIDDEN REASONING INTERNAL]\n{hidden_reasoning}\n\n"
            "Now provide a clear and concise final answer for the user request."
        )
        answer_messages = context + [
            {"role": "user", "content": answer_instructions},
        ]
        return generate(answer_messages, max_tokens=visible_token_limit)

    def _check_answer(self, context: List[Dict[str, str]], user_request: str, reasoning: str, answer: str) -> tuple[bool, str]:
        """
        Use the model to check if the answer satisfies reasoning requirements.
        Returns a feedback report with [[ACCEPTED]] or [[REVISE]].
        """

        verification_prompt = f"""
                                [CHECK ANSWER TASK]
                                Conversation context:
                                {context}

                                User request:
                                {user_request}

                                Reasoning steps:
                                {reasoning}

                                Candidate Answer:
                                {answer}

                                Instructions:
                                - For each reasoning step, check if it is reflected in the answer.
                                - Verify that the answer directly addresses the latest user request.
                                - If all requirements are satisfied, write a short report and end with [[ACCEPTED]].
                                - If not, explain what is missing and end with [[REVISE]].
                                """

        check_messages = context + [{"role": "user", "content": verification_prompt}]
        feedback_report = generate(check_messages)

        if "[[ACCEPTED]]" in feedback_report:
            return True, feedback_report
        else:
            return False, feedback_report

    def answer(self, messages: List[Dict[str, str]], visible_token_budget: int = 100) -> Dict:
        reasoning_history = []
        feedback_reports = []
        answer = None
        feedback = None

        # Split into context and current request
        context, user_request = self._split_context_and_request(messages)

        print(f"Current context: {context}")
        print(f"Current user input: {user_request}\n")

        for attempt in range(self.reasoning_depth):
            print(f"==> Reasoning step {attempt}: ")

            reasoning = self._generate_reasoning(context, user_request, feedback)
            reasoning_history.append(reasoning)

            print(f"--> Reasoning output step {attempt}:\n{reasoning}")

            answer = self._generate_answer(context, user_request, reasoning, visible_token_limit=visible_token_budget)

            print(f"--> LLM answer step {attempt}:\n{answer}")

            is_good, feedback = self._check_answer(context, user_request, reasoning, answer)
            feedback_reports.append(feedback)

            print(f"--> Answer accepted: {is_good}")
            print(f"--> Feedback: {feedback}")
            print("===========================================\n\n")

            if is_good:
                break

        # Build OpenAI-style response
        response = {
            "id": "chatcmpl-wrapper-001",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # would require token counting if needed
                "completion_tokens": 0,
                "total_tokens": 0
            },
            # Extra keys
            "reasoning_history": reasoning_history,
            "feedback_reports": feedback_reports
        }

        return response


# === Example usage ===
if __name__ == "__main__":
    wrapper = ReasoningWrapper(model="gpt-4.1-mini", reasoning_depth=3)

    conversation = [
        {"role": "user", "content": "Hi, can you help me with a math question?"},
        {"role": "assistant", "content": "Of course! What’s the problem?"},
        {"role": "user", "content": "Which is the largest prime number under 20?"}
    ]

    '''conversation = [
        {"role": "user", "content": "Which is the largest prime number under 22?"}
    ]'''

    '''conversation = [
        {"role": "user", "content": "Write a python script that outputs random numbers as fancy images."}
    ]'''

    '''conversation = [
        {"role": "user", "content": "Write a python script that generates a funny image and displays it."}
    ]'''

    result = wrapper.answer(conversation, visible_token_budget=None)

    print("=== API-LIKE OUTPUT ===")
    print(json.dumps(result, indent=2))
