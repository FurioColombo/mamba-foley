import requests

def notify_telegram(message, config, verbose=False):
    tg_config = config.telegram if hasattr(config, 'telegram') else config
    apiToken = tg_config.apiToken
    chatID = tg_config.chatID
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'
    
    if apiToken != '' and chatID != '':
        try:
            response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
            if verbose:
                print(f'Telegram message response: {response}.')
            if response.ok is not True:
                print(f'TELEGRAM NOTIFICATION ERROR: response is not OK: {str(response)}.')
        except Exception as e:
            print('Telegram notification Exception:', e)
