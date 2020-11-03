import os, sys, re

def print_usage():
    print ("Usage: %s <golden_stdout.txt> <stdout.txt>" % sys.argv[0])

class Box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def __str__(self):
        return ("Box <%f, %f, %f, %f>" % (self.x, self.y, self.w, self.h))

    def getArea(self):
        return self.w * self.h

    def overlapW(self, box):
        l1 = self.x - self.w/2
        r1 = self.x + self.w/2
        l2 = box.x - box.w/2
        r2 = box.x + box.w/2
        left = l1 if l1 > l2 else l2
        right = r1 if r1 < r2 else r2
        return right - left

    def overlapH(self, box):
        b1 = self.y - self.h/2
        t1 = self.y + self.h/2
        b2 = box.y - box.h/2
        t2 = box.y + box.h/2
        bottom = b1 if b1 > b2 else b2
        top = t1 if t1 < t2 else t2
        return top - bottom

    def intersect(self, box):
        w = self.overlapW(box)
        h = self.overlapH(box)
        if (w < 0 or h < 0): return 0
        else: return w * h

    def insersectPercent(self, box):
        return self.intersect(box) / self.getArea()
    
class Detection:    
    def __init__(self, data):
        # Class,X,Y,W,H,Objectness,Prob;
        self._class = int(data[0])
        x = float(data[1])
        y = float(data[2])
        w = float(data[3])
        h = float(data[4])
        self.box = Box(x, y, w, h)
        self.obj = float(data[5])
        self.prob = float(data[6])
    
    def __str__(self):
        return ("Detection <Class: %d, Obj: %f, Prob: %f, %s>" % (self._class, self.obj, self.prob, self.box))

def check_det(gdet, det):
    errors = []
    if (gdet._class != det._class):
        errors.append("Classes are different (%d x %d)" % (gdet._class, det._class))
    intersect = gdet.box.insersectPercent(det.box)
    if (intersect < 0.5):
        errors.append("Boxes intersection is less than 50%% (%2.1f%%)" % (intersect*100))
    return ", ".join(errors)

def check_frame(gdets, dets):
    pattern = re.compile("(\d+),(\d\.\d+),(\d\.\d+),(\d\.\d+),(\d\.\d+),(\d\.\d+),(\d\.\d+);")

    gdets = map(Detection, pattern.findall(gdets))
    dets = map(Detection, pattern.findall(dets))

    resp = ""
    
    if (len(gdets) != len(dets)):
        resp = "Amount of detections is different (%d x %d);" % (len(gdets), len(dets))
    else:
        for i, (gdet, det) in enumerate(zip(gdets, dets)):
            checkDet = check_det(gdet, det)
            if (checkDet != ""):
                resp += "Detection %d: %s; " % (i, checkDet)

    return resp

def main():
    if len(sys.argv) >= 3:
        gout = open(sys.argv[1])
        out = open(sys.argv[2])

        gdets = gout.readline()
        dets = out.readline()
        i = 0

        while gdets:
            checkFrame = check_frame(gdets, dets)
            if (checkFrame != ""):
                print("Frame %d => %s" % (i, checkFrame))
            gdets = gout.readline()
            dets = out.readline()
            i += 1
    else:
        print_usage()

if __name__ == "__main__":
    main()