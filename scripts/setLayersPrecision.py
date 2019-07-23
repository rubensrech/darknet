import os.path
import sys
import string

def readCfg(file):
    file = open(file)
    sections = {}
    unique = 0
    for line in file.readlines():
        line = line[:-1]
        try:
            if line:
                if line[0] == '#': continue
                if line[0] == '[':
                    section = line[1:-1] + str(unique)
                    if not section in sections:
                        sections[section] = {}
                    unique += 1
                elif line[0].isalpha():
                    (key, val) = line.split('=', 1)
                    sections[section][key] = val
                else: 
                    print("Error on line " + line)
        except Exception as e:
            print("Error: " + str(e) + " on line " + line)
    return sections

def setLayersReal(cfg, layers, realToBeSet):
    layerIndex = 0
    currReal = realToBeSet
    otherReal = 32 if realToBeSet == 16 else 16

    for tsec in cfg:
        sec = tsec.rstrip(string.digits)

        if sec == "net": continue

        if len(layers) > 0 and int(layers[0]) == layerIndex:
            currReal = realToBeSet
            layers.pop(0)
        else:
            currReal = otherReal

        if 'real' in cfg[tsec]:
            cfg[tsec]['real'] = str(currReal)
        
        layerIndex += 1

def saveCfg(cfg, file):
    f = open(file, "w")

    layerIndex = 0
    for tsec in cfg:
        sec = tsec.rstrip(string.digits)

        if sec != "net":
            f.write("# layer " + str(layerIndex) + "\n")
            layerIndex += 1

        f.write("[" + sec + "]\n")

        for key,val in cfg[tsec].items():
            f.write(key + "=" + val + "\n")

        f.write("\n\n")

    f.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python %s <cfgfile>" % sys.argv[0])
        exit(0)

    cfgfile = sys.argv[1]
    
    cfg = readCfg(cfgfile)
    
    print("1 - Set FLOAT layers")
    print("2 - Set HALF layers")
    action = int(input("Choose: "))

    layers = input("Enter %s layers (comma-separated): " % { 1: "FLOAT", 2: "HALF" }[action])
    layersList = layers.split(",")

    setLayersReal(cfg, layersList, { 1: 32, 2: 16 }[action])

    saveCfg(cfg, cfgfile)

if __name__ == "__main__":
    main()