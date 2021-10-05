import json
import app.SlotFilling.config as sl_cfg
import inference
import logging

sl_infer = inference.SlotFillingInference(sl_cfg.SL_MODEL_PATH, sl_cfg.DEVICE)


def predict(event, context):
    try:
        body = json.loads(event['body'])
        context.log(body)
        sentence = body['log']
        sentence = sentence.strip()
        words = sentence.split()
        words = [words]
        preds = sl_infer.predict(words)
        context.log(preds)
        logging.info(f"prediction: {preds}")

        return {
            "statusCode": 200,
            "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
            "body": json.dumps({"prediction": preds['class']})
        }
    except Exception as e:
        logging.error(e)
        return {
            "statusCode": 500,
            "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
            "body": json.dumps({"error": repr(e)})
        }

def predict_off(sentence):
    sentence = sentence.strip()
    words = sentence.split()
    words = [words]
    preds = sl_infer.predict(words)
    return preds

# def main():
#     sentence = 'từ trường đại học bách khoa thành phố hồ chí minh đến ngã tư bảy hiền bị kẹt xe do có quá nhiều xe vận tốc đạt được khoảng 60 ki lô mét trên giờ'
#     print(predict_off(sentence))

# if __name__ == '__main__':
#     main()