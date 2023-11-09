import os
import dotenv
import requests
dotenv.load_dotenv()
SlackURL = "https://slack.com/api/chat.postMessage"
LineURL = "https://notify-api.line.me/api/notify"

class SlackClient:
    def __init__(self) -> None:
        self._token = os.getenv('SlackToken')
        self._channel = os.getenv('SlackChannel')
        if self._token is None or self._channel is None:
            raise ValueError("SlackToken or SlackChannel does not exist.")
        self._headers = {"Authorization": "Bearer "+ self._token}

    def send_text(self, text: str) -> None:
        data  = {
        'channel': self._channel,
        'text': text
        }
        r = requests.post(SlackURL, headers=self._headers, data=data)
        print("return ", r.json())

class LineClient:
    def __init__(self) -> None:
        self._token = os.getenv('LineToken')
        if self._token is None:
            raise ValueError("LineToken does not exist.")
        self._headers = {"Authorization": "Bearer "+ self._token}

    def send_text(self, text: str) -> None:
        payload  = {
        'message': text
        }
        r = requests.post(LineURL, headers=self._headers, params=payload )
        print("return ", r.json())
    
    def send_dict(self, data: dict):
        text = "\n".join([f"{k}: {v}" for k, v in data.items()])
        self.send_text(text)

if __name__ == '__main__':
    client = LineClient()
    client.send_text("test")