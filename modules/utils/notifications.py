import requests

def notify_telegram(message, verbose=False):
    apiToken = '7133523436:AAHSM_44T1MFKLhkHxm5myZ7u0o1IE0Xc6Q'
    chatID = '275894223'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        if verbose:
            print(f'Telegram message response: {response}.')
        if response.ok is not True:
            print(f'TELEGRAM NOTIFICATION ERROR: response is not OK: {str(response)}.')
    except Exception as e:
        print('Telegram notification Exception:', e)
