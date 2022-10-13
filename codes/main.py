from abc import abstractclassmethod
import os
import numpy as np
import cv2
import threading
import mediapipe as mp
import collections
import time
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from tensorflow.keras.models import load_model


class MaskDetectBase(object):
  def __init__(self):
    pass
  # end

  @abstractclassmethod
  def maskDetect(self, img):
    pass
  # end
  
  def relu(self, arr):
    return [x * (x > 0) for x in arr]

class MaskDetect(MaskDetectBase):
  def __init__(self):
    super().__init__()
    self.modelpth = os.getcwd() + "/weights/mask_detector.h5"
    self.model = load_model(self.modelpth)
    self.threshold_unstable = 0.2
  
  def maskDetect(self, img, bboxes):
    frameCopy = img.copy()
    mask_statuses = {}
    for ids in bboxes:
      bbox = bboxes[ids]
      (x, y, w, h) = bbox
  
      (x, y, w, h) = self.relu([x+5, y-5, w+15, h-15])
      croppedFace = frameCopy[y:h, x:w]

      face = cv2.cvtColor(croppedFace, cv2.COLOR_BGR2RGB)
      face = cv2.resize(face, (224, 224))
      face = 2.*(face - np.min(face))/np.ptp(face)-1
      face = np.expand_dims(face, axis=0)
      
      (mask, withoutMask) = self.model.predict(face)[0]
      if abs(mask - withoutMask) > self.threshold_unstable: #0.2
        text = [1 if (mask > withoutMask) else 0][0]
      else:
        text = 2
        
      mask_statuses[ids] = [text]
    return frameCopy, mask_statuses

class MaskDetect_Converted_ONNX(MaskDetectBase):
  def __init__(self):
    super().__init__()
    model_root = os.getcwd() + "/weights/mask_detect_binary.onnx"
    self.detector = ort.InferenceSession(model_root, providers=['CUDAExecutionProvider'])
    self.input_name = self.detector.get_inputs()[0].name
    self.output_name = self.detector.get_outputs()[0].name
    self.threshold_unstable = 0.2
      
  def maskDetect(self, img, bboxes):
    frameCopy = img.copy()
    mask_statuses = {}
    stat = 2
  
    for ids in bboxes:
      bbox = bboxes[ids]
      (x, y, w, h) = bbox
      (x, y, w, h) = self.relu([x+5, y-5, w+15, h-15])
      frame = frameCopy[y:h, x:w]  #[y-5:h-10, x+5:w+10]
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = cv2.resize(frame, (224, 224))
      frame = 2.*(frame - np.min(frame))/np.ptp(frame)-1 # -1 ile 1 arasında normalizasyon
      frame = np.expand_dims(frame, axis=0)
      
      #Inference
      scores = self.detector.run(None, {self.input_name: frame.astype(np.float32)})[0][0]
      mask, withoutMask = scores
      dist = abs(mask - withoutMask)

      if dist > self.threshold_unstable: #0.2
        stat = [1 if (mask > withoutMask) else 0][0]
      else:
        stat = 2
  
      mask_statuses[ids] = [stat]
  
    return frameCopy, mask_statuses


#Test Aşamasında
class MaskDetect_ONNX(MaskDetectBase):
  def __init__(self):
    super().__init__()
    model_root = os.getcwd() + "/weights/mask_detect_correctly.onnx"
    self.process_transform = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
    self.detector = ort.InferenceSession(model_root, providers=['CUDAExecutionProvider'])
    self.input_name = self.detector.get_inputs()[0].name
    self.output_name = self.detector.get_outputs()[0].name
    self.threshold_unstable = 0.41
    
  def mask_stats(self, data:int):
    statuses = {
        0: 1, #maskeli
        1: 3, #yanlis kullanim
        2: 0, #maskesiz
    }
    return statuses.get(data, 2)
  
  def softmax(self, scores):
        e_x = np.exp(scores - np.max(scores))
        return e_x / e_x.sum()
  
  def maskDetect(self, img, bboxes):
    frameCopy = img.copy()
    mask_statuses = {}
    for ids in bboxes:
      bbox = bboxes[ids]
      (x, y, w, h) = bbox
      (x, y, w, h) = self.relu([x+5, y-5, w+15, h-15])
      frame = frameCopy[y:h, x:w]
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = Image.fromarray(frame)
      frame = self.process_transform(frame)
     
      frame = frame.numpy()
      frame = 2.*(frame - np.min(frame))/np.ptp(frame)-1 # -1 ile 1 arasında normalizasyon
      frame = np.expand_dims(frame, axis=0)
      
      #Inference
      scores = self.detector.run(None, {self.input_name: frame.astype(np.float32)})[0][0]
    
      #PostProcess
      scores = self.softmax(scores)
      print(scores)
      mask = scores[0]
      incorrect_masked = scores[1]
      unmasked = scores[2]
      
      idx = np.argmax(scores)
      element = scores[idx]
      scores = np.delete(scores, idx)
      arr = abs(scores - element) < self.threshold_unstable
      enough = (np.matrix.dot(np.transpose(arr), arr))
      print(enough)
      if not enough:
        text = self.mask_stats(idx)
      else:
        text = 2
      mask_statuses[ids] = [text]

    return frameCopy, mask_statuses

class covid_mask_detection(object):
  def __init__(self, image_path=None, source=0, algorithm_number = 1) -> None:
    self.frame = []
    self.image_path = image_path
    self.algorithm_number = algorithm_number
    self.stop_crit = False
    self.source = source
    self.is_file = True if self.image_path!=None else False
    if self.is_file==True:
      try:
        frame = cv2.imread(self.image_path)
        self.frame = frame
      except Exception as e:
        print("Dosya Okunurken Hata", e)
        raise FileNotFoundError
    else:
      self.cap = cv2.VideoCapture(self.source)
      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      self.cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)
      self.cap.set(cv2.CAP_PROP_FPS, 30)
      self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H265'))
      self.t = threading.Thread(target = self.updateFrame)
      self.t.start()

    self.iter_count = 0
    self.MIN_STAT = 3
    self.MIN_FRAME = self.MIN_STAT + 4
    self.analysis_arr = collections.defaultdict(list)
    mp_face_detection = mp.solutions.face_detection
    self.face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.35)
    
    self.mask_detector_names = {
      0: MaskDetect,
      1: MaskDetect_Converted_ONNX,
      2: MaskDetect_ONNX
    }
    self.mask_detector = self.mask_detector_names[self.algorithm_number]()
  
  def updateFrame(self):
    while self.cap.isOpened():
      success, frame = self.cap.read()
      if self.stop_crit:
        self.cap.release()
        raise StopIteration
      if not success:
        continue
      self.frame = frame
      time.sleep(0.0001)
  
  def mask_stats(self, data:int):
    statuses = {
        0: ["Maskesiz", (0,0,255)],
        1: ["Maskeli", (0,255,0)],
        2: ["Kararsiz", (255,255,0)],
        3: ["Yanlis_Kullanim", (110,0,250)]
    }
    return statuses.get(data, ["Bilinmeyen", (0,0,0)])
  
  def __iter__(self):
    return self
    
  def __next__(self):
    self.iter_count = 1 + self.iter_count
    cframe = []
    mask_statuses = "Bilinmeyen"
    
    if self.stop_crit:
      self.cap.release()
      self.t.join()
      raise StopIteration
    else:
      frame = []
      bboxes = []
      if self.frame != []:
        self.frame = cv2.resize(self.frame, (1280,720))  # Bilgisayar ekranından taşmaması için yeniden boyutlandırma(isteğe bağlı)
        frame, bboxes = self.face_detect(self.frame)
        cframe, mask_statuses = self.mask_detector.maskDetect(frame, bboxes)
        
        #Yöntem-1
        #Analiz etmeden çiz
        #cframe = self.draw_statuses(frame, bboxes, mask_statuses)

        #Yöntem-2
        #Analiz ettikten sonra çiz(id switching olacaktır çünkü bu projede sisteme daha fazla yük bindiren deepsort veya face recognition kullanılmamaktadır.)
        for ids in mask_statuses:
          self.analysis_arr[ids].append(mask_statuses[ids])
        if self.iter_count%self.MIN_FRAME == 0:
          self.iter_count = 0
          stable_analysis = self.analysis(self.analysis_arr, self.MIN_STAT)
          cframe = self.draw_statuses(frame, bboxes, stable_analysis)
          self.analysis_arr = collections.defaultdict(list)
          
      return cframe, mask_statuses
      
  def analysis(self, arr, min_stat = 3):
    arrays = {}
    for ids in arr:
      if len(arr[ids]) > min_stat:
        arrays[ids] = self.find_most_frequency_ones_along_y_axis(arr[ids])
    return arrays
  
  def find_most_frequency_ones_along_y_axis(self, arr):
    """ Ekranda görülen İlgili indisteki kişi için toplanan değerlerin ortalaması alınarak çıkarımda bulunulur.
    [[1,2,3],   |                 [[True],   |
      [1,2,0],  |---> [1, 2, 0]    [False],  |---> False
      [0,1,0]]  |                  [False]]  |
    """
    arr = np.array(arr)
    uniq, indices = np.unique(arr, return_inverse=True)
    most_freq = uniq[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(arr.shape), None, np.max(indices) + 1), axis=0)]
    return most_freq

  def relu(self, arr):
    return [x * (x > 0) for x in arr]
    
  def xywh2xyzt(self, bbox):
    x, y, w, h = bbox
    return (x, y, x+h, y+w)

  def stop_iter(self):
    self.stop_crit = True
  
  def draw_statuses(self, frame, bboxes, statuses):
    copyFrame = frame.copy()
    copyFrame.flags.writeable = True
    if len(bboxes) > 0:
      for ids in bboxes:
        bbox = bboxes[ids]
        (x, y, z, t) = bbox
        stat = statuses.get(ids, [2])
        settings = self.mask_stats(stat[0])
        print("settings", settings)
        (text, color) = settings
        copyFrame = cv2.rectangle(copyFrame, (x, y), (z, t), color, 2)
        copyFrame = cv2.putText(copyFrame, str(text), (x,y-5), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color, 1)
      
    return copyFrame 
  
  def face_detect(self, image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Frame i RGB formatına dönüştür.
    results = self.face_detection.process(image)   #RGB frame i ilgili modele vererek yüzün koordinatlarını al
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = True

    bboxes = {}
    if results.detections:
      for i, detection in enumerate(results.detections):
        bbox = detection.location_data.relative_bounding_box

        a, b, _ = image.shape
        y = a*bbox.ymin
        w = a*bbox.height
        x = b*bbox.xmin
        h = b*bbox.width
        xx, yy, ww, hh = np.array((x, y, w, h)).astype(np.int)

        #Negatif Değerleri 0 yaparak, çerçeve dışına çıkılmasını engeller.
        (xx, yy, ww, hh) = self.relu([xx, yy, ww, hh])
        (x, y, z, t) = self.xywh2xyzt((xx, yy, ww, hh))

        bboxes[i] = (x, y, z, t)
    return image, bboxes

if __name__ == "__main__":
  detector = covid_mask_detection(image_path=None, source = 0, algorithm_number=1) #algorithm_number :(0,1,2)

  for cframe, mask_statuses in detector:
    if cframe != []:
      cv2.imshow("test", cframe)
    
    if cv2.waitKey(3) & 0xFF == ord("q"):
      detector.stop_iter()