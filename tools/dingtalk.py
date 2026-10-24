import requests
import json
import time
import hmac
import hashlib
import base64
import urllib.parse
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import os
from omegaconf import OmegaConf
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def dingtalk(msg, access_token, secret):    
    timestamp = str(round(time.time() * 1000))
    secret_enc = secret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    url = f'https://oapi.dingtalk.com/robot/send?access_token={access_token}&timestamp={timestamp}&sign={sign}'
    payload = {'text': {'content': msg}, 'msgtype': 'text'}
    r = requests.post(url, json=payload, verify=False)
    print('dingtalk:', r.text)

def round_floats(o):
    if isinstance(o, float): return round(o, 3)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o

def get_metric(root_path):
    tf_event = sorted(root_path.glob('events.out.tfevents.*'), 
                      key=os.path.getmtime, reverse=True)[0] # the latest
    ea = event_accumulator.EventAccumulator(str(tf_event))
    ea.Reload()
    metric = {}
    for key in ea.scalars.Keys():    
        if 'valid' in key:
            metric[key[6:]] = ea.scalars.Items(key)[-1].value
    return metric

def dingtalk_metric(root_dir):
    root_path = Path(root_dir)
    test_results_path = root_path.joinpath('test_results.json')
    token_conf = OmegaConf.load(conf['token_conf'])
    if test_results_path.exists():
        metric = json.load(open(test_results_path))
    else:
        metric = get_metric(root_path)
    msg = root_path.parent.name + '/' + root_path.name
    msg = msg + '\n' + json.dumps(round_floats(metric))
    try:
        dingtalk(msg, **token_conf['dingtalk'])
    except Exception as e:
        print('dingtalk error:', e)

def test_dingtalk(conf):
    r = requests.request('GET', 'http://www.baidu.com', verify=False)
    token_conf = OmegaConf.load(conf['token_conf'])
    dingtalk('消息测试', **token_conf['dingtalk'])
    
if __name__ == '__main__':
    conf = OmegaConf.create({
         'cmd': 'test_dingtalk',
         'token_conf': 'token.yaml'
         })
    conf.merge_with_cli()    
    eval(conf.cmd)(conf)
    