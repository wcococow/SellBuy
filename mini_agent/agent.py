class LLMCore:
    def __init__(self, client):
        self.client = client  # OpenAI lives here

    def generate(self, prompt):
        return self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
