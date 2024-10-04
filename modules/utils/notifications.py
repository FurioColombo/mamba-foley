import requests

def notify_telegram(message, config, verbose=False):
    tg_config = config.telegram if hasattr(config, 'telegram') else config
    if hasattr(config, 'apiToken') and hasattr(config, 'chatID') and hasattr(config, 'apiURL'):
        api_token = tg_config.apiToken
        chat_id = tg_config.chatID
        api_url = f'https://api.telegram.org/bot{api_token}/sendMessage'
        
        if api_token != '' and chat_id != '':
            try:
                response = requests.post(api_url, json={'chat_id': chat_id, 'text': message})
                if verbose:
                    print(f'Telegram message response: {response}.')
                if response.ok is not True:
                    print(f'TELEGRAM NOTIFICATION ERROR: response is not OK: {str(response)}.')
            except Exception as e:
                print('Telegram notification Exception:', e)
