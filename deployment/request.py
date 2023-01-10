import requests


input_json = {'age': 3.4657359027997265,
                'emp.var.rate': 1.4,
                'cons.conf.idx': 42.7,
                'euribor3m': 4.0,
                'nr.employed': 8.561803202334707,
                'campaign': 16.0,
                'pdays': 0.0,
                'cons.price.idx': 4.542422061134141,
                'default': 1.0,
                'housing': 0.0,
                'loan': 0.0,
                'contact': 1.0,
                'previous': 0.0,
                'poutcome': 0.0,
                'marital_te': 0.10323140316634136,
                'education_te': 0.13720815521210128,
                'job_te': 0.08516483516483517,
                'month_sin': -0.5000000000000001,
                'month_cos': -0.8660254037844386,
                'day_sin': -0.9749279121818235,
                'day_cos': -0.2225209339563146}


url = 'http://localhost:8080/predict'
response = requests.post(url,json=input_json)

print(response.json())
