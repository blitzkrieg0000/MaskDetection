from abc import abstractclassmethod
from concurrent.futures import thread
from logging import PlaceHolder
import os
import numpy as np
import cv2
import threading
import mediapipe as mp
import collections
import time
import onnxruntime as ort
from torchvision import transforms
from tensorflow.keras.models import load_model
from tkinter import *
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from playsound import playsound
import platform
OS = platform.system()

#UI------------------------------------------------>

class Mousewheel_Support(object):    
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, root, vertical_factor=2):
        
        self._active_area = None
        if isinstance(vertical_factor, int):
            self.vertical_factor = vertical_factor
        else:
            raise Exception("Değer tam sayı olmalıdır.")

        if OS == "Linux" :
            root.bind_all('<4>', self._on_mousewheel,  add='+')
            root.bind_all('<5>', self._on_mousewheel,  add='+')
        else:
            root.bind_all("<MouseWheel>", self._on_mousewheel,  add='+')

    def _on_mousewheel(self, event):
        if self._active_area:
            self._active_area.onMouseWheel(event)

    def _mousewheel_bind(self, widget):
        self._active_area = widget

    def _mousewheel_unbind(self):
        self._active_area = None

    def add_support_to(self, widget=None, yscrollbar=None, what="units", vertical_factor=None):
        if yscrollbar is None:
            return

        if yscrollbar is not None:
            vertical_factor = vertical_factor or self.vertical_factor
            yscrollbar.onMouseWheel = self._make_mouse_wheel_handler(widget,'y', self.vertical_factor, what)
            yscrollbar.bind('<Enter>', lambda event, scrollbar=yscrollbar: self._mousewheel_bind(scrollbar) )
            yscrollbar.bind('<Leave>', lambda event: self._mousewheel_unbind())

        if widget is not None:
            if isinstance(widget, list) or isinstance(widget, tuple):
                list_of_widgets = widget
                for widget in list_of_widgets:
                    widget.bind('<Enter>',lambda event: self._mousewheel_bind(widget))
                    widget.bind('<Leave>', lambda event: self._mousewheel_unbind())

                    widget.onMouseWheel = yscrollbar.onMouseWheel
            else:
                widget.bind('<Enter>',lambda event: self._mousewheel_bind(widget))
                widget.bind('<Leave>', lambda event: self._mousewheel_unbind())

                widget.onMouseWheel = yscrollbar.onMouseWheel

    @staticmethod
    def _make_mouse_wheel_handler(widget, orient, factor = 1, what="units"):
        view_command = getattr(widget, orient+'view')
        
        if OS == 'Linux':
            def onMouseWheel(event):
                if event.num == 4:
                    view_command("scroll",(-1)*factor, what)
                elif event.num == 5:
                    view_command("scroll",factor, what) 
                
        elif OS == 'Windows':
            def onMouseWheel(event):        
                view_command("scroll",(-1)*int((event.delta/120)*factor), what) 
        
        elif OS == 'Darwin':
            def onMouseWheel(event):        
                view_command("scroll",event.delta, what)
        
        return onMouseWheel
#end

class ScrollableCanvas(tk.Canvas):
  def __init__(self, container, *args, **kwargs):
    super().__init__(container, *args, **kwargs)
    self.columns=0
    
    #Scrolable Area
    self.scrollable_canvas = tk.Canvas()
    self.scrollable_canvas.bind("<Configure>", lambda e: self.configure(scrollregion=self.bbox("all")))

    #Scrollbar
    self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.yview)
    Mousewheel_Support(self).add_support_to(self, yscrollbar=self.scrollbar)
    self.scrollbar.pack(side="right", fill="y")
    self.configure(yscrollcommand=self.scrollbar.set)

    #Add Scrollable Area To ThisClassCanvas
    self.create_window((0, 0), window=self.scrollable_canvas, anchor=N+W)
    
    #Regrid If ThisClassCanvas' Area changes
    self.bind('<Configure>', self.regrid)
    self.bind("<<regrid>>", self.regrid)
    
    #Insert ThisClassCanvas in parent by "fill" and "expand"
    self.pack(side="left", fill="both", expand=True)

  def regrid(self, event=None):
    grid_width = self.winfo_width()
    slaves = self.scrollable_canvas.grid_slaves()
  
    if len(self.scrollable_canvas.children) > 1:
      slaves_width = slaves[0].winfo_width()
      
      if slaves_width == 1:
        return
      
      cols = (grid_width // slaves_width)

      if (cols == 0): #(cols == self.columns) | 
        return

      for i, slave in enumerate(reversed(slaves)):
        print(i, slave)
        try:
          slave.grid_forget()
        except:
          pass
        slave.grid(row=i // cols, column=i % cols)
        slave.update()

      self.columns = cols
      self.scrollable_canvas.update()
      self.scrollbar.update()
      self.configure(scrollregion=self.bbox("all"))
    else:
      return
#end

class addFrame(tk.Canvas):
  def __init__(self, master=None, add_image=None, **kwargs):
    tk.Canvas.__init__(self, master, bg="#400354", bd=2, width=100, height=150, relief=tk.RAISED, **kwargs)
    self.img = cv2.resize(add_image, (100, 150))
    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
    self.pimg = Image.fromarray(self.img)
    self.ptkimg = ImageTk.PhotoImage(self.pimg)
    self.create_image(0,0, image=self.ptkimg, anchor=N+W)
    self.grid()
#end

class UI(object):
  def __init__(self):
    #MAIN CANVAS
    self.root = Tk()
    self.root.title("Maskeli-Maskesiz")
    self.root.geometry("1200x600+300+200")
    self.root.resizable(width=True, height=True)

    self.threshold = 0.5
    
    self.pimg = []
    self.canvas = Canvas(self.root, width=640, height=480)
    self.canvas.pack(side="left", fill="both", expand=True)
    
    self.video_canvas = Canvas(self.root, bg="#757575")
    self.video_canvas.place(relx = 0.01, rely=0.01, relheight=0.98, relwidth=0.6)
    self.video_canvas.bind("<Configure>", self.resizer)
    
    self.temp_canvas = Canvas(self.root, bg="#000000")
    self.temp_canvas.place(relx = 0.62, rely=0.1, relheight=0.89, relwidth=0.37, anchor=N+W)
    
    self.temp_scroll_canvas = ScrollableCanvas(self.temp_canvas)
    
    self.menu_canvas = Canvas(self.root, bg="#8c8c8c")
    self.menu_canvas.place(relx = 0.62, rely=0.01, relheight=0.09, relwidth=0.37, anchor=N+W)
    
    self.delete_button = ttk.Button(self.menu_canvas, text ="Temizle", command = self.delete_temp_canvas)
    self.menu_canvas.create_window(0, 0, window=self.delete_button, anchor=N+W)
    #self.delete_button.pack(expand = True)
  
    self.threshold_entry = tk.Entry(self.menu_canvas)
    self.menu_canvas.create_window(105, 30, window=self.threshold_entry, anchor=N+W, width=35) #,  height=10
  
    self.threshold_button = ttk.Button(self.menu_canvas, text ="Set Threshold", command = self.set_threshold)
    self.menu_canvas.create_window(0, 30, window=self.threshold_button, anchor=N+W)

    self.threshold_label = tk.Label(self.menu_canvas, text=f"value:{self.threshold} default:0.5", font=('helvetica', 10, 'bold'))
    self.menu_canvas.create_window(250, 40, window=self.threshold_label)


  def set_threshold(self):
    threshold = self.threshold_entry.get()
    
    if isinstance(threshold, str):
      threshold = float(threshold)
      
    if threshold < 0 or threshold > 1:
      return
    
    self.threshold = threshold
    self.threshold_label.config(text=f"value:{self.threshold} default:0.5")
    
  def delete_temp_canvas(self):
    for child in self.temp_scroll_canvas.scrollable_canvas.winfo_children():
      child.destroy() #if child != ".!canvas3.!scrollablecanvas.!scrollbar":
  
  def resizer(self, e=None):
    self.tkresized_image = []
    if self.pimg != []:
      self.resized_image = self.pimg.resize((e.width,e.height), Image.ANTIALIAS)
      self.tkresized_image = ImageTk.PhotoImage(self.resized_image)
      self.video_canvas.create_image(0,0, image=self.tkresized_image, anchor=NW)
    else:
      return
  
  def updateVideoScreen(self, img = []):
    self.ww = self.video_canvas.winfo_width()
    self.hh = self.video_canvas.winfo_height()
    if self.ww == 1 | self.hh == 1:
      self.ww = 1920
      self.hh = 1080
    self.img = cv2.resize(img, (self.ww, self.hh))
    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
    self.pimg = Image.fromarray(self.img)
    self.video_image = ImageTk.PhotoImage(self.pimg)
    self.video_canvas.create_image(0,0, image=self.video_image, anchor=N+W)
    self.video_canvas.update()
  
  def addTemp(self, add_image):
    addFrame(self.temp_scroll_canvas.scrollable_canvas, add_image=add_image)
#end


#Algorithm----------------------------------------->
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
#end

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
#end

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
#end

class MaskDetect_ONNX(MaskDetectBase):
  "Deneysel"
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

      mask = scores[0]
      incorrect_masked = scores[1]
      unmasked = scores[2]
      
      idx = np.argmax(scores)
      element = scores[idx]
      scores = np.delete(scores, idx)
      arr = abs(scores - element) < self.threshold_unstable
      enough = (np.matrix.dot(np.transpose(arr), arr))

      if not enough:
        text = self.mask_stats(idx)
      else:
        text = 2
      mask_statuses[ids] = [text]

    return frameCopy, mask_statuses
#end

class covid_mask_detection(object):
  def __init__(self, image_path=None, source=0, algorithm_number=1) -> None:
    self.frame = []
    self.image_path = image_path
    self.algorithm_number = algorithm_number
    self.stop_crit = False
    self.threshold = 0.5
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
      self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2) #cv2.CAP_V4L2  hata vermesine karşın
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
    self.mp_face_detection = mp.solutions.face_detection
    self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=self.threshold) #0.35 Mediapipe
    
    self.mask_detector_names = {
      0: MaskDetect,
      1: MaskDetect_Converted_ONNX,
      2: MaskDetect_ONNX
    }
    self.mask_detector = self.mask_detector_names[self.algorithm_number]()
  
  def update_threshold(self, threshold):
    self.threshold = threshold
    self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=threshold)
  
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
    stable_analysis = {0:[2]}
    if self.stop_crit:
      self.cap.release()
      self.t.join()
      raise StopIteration
    else:
      frame = []
      bboxes = []
      if len(self.frame)>0:
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
          
      return self.frame, cframe, stable_analysis, bboxes
      
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
#end


if __name__ == "__main__":
  source = 0 #"http://192.168.1.111:8080/video"
  detector = covid_mask_detection(image_path=None, source=source, algorithm_number=1) #algorithm_number :(0,1,2)
  
  APP = UI()
  APP.threshold = detector.threshold
  
  def sound():
    playsound("assets/sounds/pizzicato.ogg")
  
  for frame, cframe, stable_analysis, bboxes in detector:

    if len(cframe)>0:
      APP.updateVideoScreen(cframe)
    
    #Threshold değişmiş ise güncelle
    if APP.threshold != detector.threshold:
      detector.update_threshold(APP.threshold)
    
    if bboxes != []:
      for x in bboxes:
        bbox = bboxes[x]

        if stable_analysis.get(x,[1])[0]==0: #Maskesiz
          (x, y, w, h) = bbox
          cropped_face = frame[y:h, x:w]
          threading.Thread(target = sound).start()
          APP.addTemp(cropped_face)
    
    APP.temp_scroll_canvas.event_generate("<<regrid>>", when="now")
    
    if cv2.waitKey(3) & 0xFF == ord("q"):
      detector.stop_iter()
    
  APP.root.mainloop()
#end