from flask import Flask

from garbage_det import * 

app = Flask(__name__)
from flask import Flask, jsonify, make_response
from flask import request

@app.route('/garbage/<path>')
def index(path):
    print(path)
    lit = t.test_function()
    return lit+path	

@app.route('/garbage/analyze', methods = ['POST'])
def analyze_aisle_image():
  #print('request.args ', request.form)
  print(request.files)
  target = 'images'
  file = request.files['cssv_file'] 
  file.save('./images/test.jpg')
  #print(os.system("python3 garbage_det.py detect --weights='./mask_rcnn_garbage_0010.h5' --image='test.jpg'"))
        
  ret_value = detect_and_save("hi", image_path='./images/test.jpg')
  if(ret_value):
    return "True"
  else:
    return "False"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)
