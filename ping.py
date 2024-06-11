import requests


def ping_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Website {url} is UP!")
        else:
            print(f"Website {url} is DOWN with status code {response.status_code}.")
    except requests.ConnectionError:
        print(f"Failed to connect to {url}. It may be DOWN.")

url = "https://funboyprojects.com"

while True:
    ping_website(url)