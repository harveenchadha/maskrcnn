from flask import Flask
import t

app = Flask(__name__)

@app.route('/garbage/<path>')
def index(path):
    print(path)
    lit = t.test_function()
    return lit+path	

@app.route('/garbage/analyze', methods = ['POST'])
def analyze_aisle_image():
   print('request.args ', request.form)
   ais_img=request.form.get('AIS_IMG', default='./ais_repo/ais1.png', type=str)
   response_text = process_pipeline(ais_img, brand, item)
   resp = make_response(json.dumps(response_text))
   resp.headers['Access-Control-Allow-Origin'] = '*'
   return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    app.run(debug=True)
