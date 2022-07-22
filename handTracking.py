from cvzone.HandTrackingModule import HandDetector
import cv2,numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
from sklearn.decomposition import PCA


pcaHand = PCA(n_components=3)

def findBestMatches(nbrs,new_row,n_neighbors=2):
    distances,indices = nbrs.kneighbors([np.array(new_row)],n_neighbors=n_neighbors)
    dx = sorted(list(zip(distances[0],indices[0])))
    print(dx)
    indi = [d[1] for d in dx]
    print(indi)
    #print(distances)
    #distances,indices = nbrs.query([np.array(new_row)],k=n_neighbors)
    return indi

def findByRadius(nbrs,new_row,radius=10.0):
    distances,indices = nbrs.radius_neighbors([np.array(new_row)],radius=radius,return_distance=True,sort_results=True)
    #distances,indices = nbrs.query([np.array(new_row)],k=n_neighbors)
    print(radius,list(zip(distances[0],indices[0])))
    return indices[0],distances[0]

def getNoteVectorsDict(fn="./data/music/note_vectors.csv",dim=-1,h=100):
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    #pca.fit(X)

    noteVectors = pd.read_csv(fn,sep=";")
    
    keys = [tuple(t) for t in (noteVectors[["midi_pitch","note_duration_numerator","note_duration_denominator","dynamics","flag_is_rest"]]).head(h).values.tolist()]
    if dim==-1:
        vals = [tuple(t) for t in (noteVectors[[col for col in noteVectors.columns if "vec_" in col]]).head(h).values.tolist()]
    else:
        pca = PCA(n_components=dim)
        vals = [tuple(t) for t in pca.fit_transform((noteVectors[[col for col in noteVectors.columns if "vec_" in col]]).head(h).values).tolist()]
        
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree').fit(vals)
    return keys, dict(zip(keys,vals)),nbrs




noteNames,nvd,nbrs = getNoteVectorsDict(dim=3,h=-1)

print(noteNames,nvd)

currentLmList = []
daumen_note,zeigefinger_note,mittelfinger_note,ringfinger_note,kleinerfinger_note = 5*[None]

d,zf,mf,rf,kf = [],[],[],[],[]

def imageInLoop():
    global cap, detector, currentLmList, pcaHand,daumen_note,zeigefinger_note,mittelfinger_note,ringfinger_note,kleinerfinger_note
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        #print(lmList1)
        currentLmList.append(lmList1)
        if len(currentLmList)==2:
            last = np.array(currentLmList[-2])
            current = np.array(lmList1)
            diff = current#-last
            pcaHand.fit(diff)
            
        if len(currentLmList)>=2:
            last = np.array(currentLmList[-2])
            current = np.array(lmList1)
            diff = current#-last
            pcadiff = pcaHand.transform(diff)
            #print(pcadiff)
            #4,8,12,16,20
            #daumen, zeigefinger,...,kleiner finger
            daumen_indices = findBestMatches(nbrs,pcadiff[4].tolist(),n_neighbors=1)
            daumen_note = noteNames[int(daumen_indices[-1])]
            
            #zeigefinger
            zeigefinger_indices = findBestMatches(nbrs,pcadiff[8].tolist(),n_neighbors=1)
            zeigefinger_note = noteNames[int(zeigefinger_indices[-1])]
            
            #mittelfinger
            mittelfinger_indices = findBestMatches(nbrs,pcadiff[12].tolist(),n_neighbors=1)
            mittelfinger_note = noteNames[int(mittelfinger_indices[-1])]

            #ringfinger
            ringfinger_indices = findBestMatches(nbrs,pcadiff[16].tolist(),n_neighbors=1)
            ringfinger_note = noteNames[int(ringfinger_indices[-1])]
            
            #kleinerfinger
            kleinerfinger_indices = findBestMatches(nbrs,pcadiff[20].tolist(),n_neighbors=1)
            kleinerfinger_note = noteNames[int(kleinerfinger_indices[-1])]
                        
            
            print("daumen =", daumen_note)
            print("zeigefinger = ",zeigefinger_note)
            print("mittelfinger = ",mittelfinger_note)
            print("ringfinger = ",ringfinger_note)
            print("kleinerfinger = ",kleinerfinger_note)
            d.append(daumen_note)
            zf.append(zeigefinger_note)
            mf.append(mittelfinger_note)
            rf.append(ringfinger_note)
            kf.append(kleinerfinger_note)
            
            
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)

            # Find Distance between two Landmarks. Could be same hand or different hands
            #length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
            # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    

import numpy as np
from getkey import getkey, keys
import random, sys, pickle
from scamp import Session, Ensemble, current_clock
from compute_knn_model import *
from music21.pitch import Pitch

#print([pitch.Pitch(midi=int(p)) for p in pitchlist])



from scamp import Session, Ensemble
from scamp._soundfont_host import get_best_preset_match_for_name

def get_general_midi_number_for_instrument_name(p,sf):
    ensemble = Ensemble(default_soundfont=sf)
    return (get_best_preset_match_for_name(p,which_soundfont=ensemble.default_soundfont)[0]).preset

def construct_ensemble(sf,std):
    global piano_clef,piano_bass, flute, strings, session
    ensemble = Ensemble(default_soundfont=sf)

    ensemble.print_default_soundfont_presets()

    return [(ensemble.new_part(p),get_best_preset_match_for_name(p,which_soundfont=ensemble.default_soundfont)) for p in std] 




   
def piano_play_note(pitch,volume,duration,isPause,instr_to_play):
    global instrs,s,tempo,instrumentName,knn_is_reverse,knn_nr_neighbors
    if isPause:
        return
    length = duration
    print("playing "+instrumentName+" [",instr_to_play,"]: ", Pitch(midi=int(pitch)),volume,duration,length)
    #print("knn_reverse = ",knn_is_reverse)
    print("knn_nr_neighbors = ", knn_nr_neighbors)
    s.instruments[instr_to_play].play_note(pitch,volume/128.0,length,blocking=True)



def append_knn_notes(instr_to_play,note):
    global knn_notes_to_play,global_counter_midi_pressed,instrs,list_nbrs_notes_octaves,knn_is_reverse,knn_nr_neighbors
    global jump
    #instr_to_play = global_counter_midi_pressed % len(instrs)
    pitch,duration,volume,isPause = note
    knn_notes_to_play[instr_to_play].append(note)   
    #play_left_hand = 1*(pitch<128//2)
    #instr_to_play = 0 #play_left_hand
    #new_row = np.array([pitch,duration,volume,isPause])
    #octaves = pitch//36
    #nbrs,notes = list_nbrs_notes_octaves[octaves]
    #if knn_nr_neighbors > 0:
    #    bm = findBestMatches(nbrs,new_row,n_neighbors=knn_nr_neighbors,reverse=knn_is_reverse)
    #    if jump:
    #        jump = False
    #        bm = [bm[-1]]    
    #    print([notes[b] for b in bm])
    #    for b in bm:
    #        pitch,duration,volume,isPause = notes[b]
    #        knn_notes_to_play[instr_to_play].append(notes[b])   
    
        
def scamp_loop():
    global inds,tracks,s,started_transcribing,knn_notes_to_play_for_marimba,knn_notes_to_play,global_counter_midi_pressed,instrs
    global d, zf,mf,rf,kf
    
   
    while True:
        #instr_to_play = global_counter_midi_pressed % len(instrs)   
        if len(d)>0:
            note = d.pop(0)
            pitch,duration,volume,isPause = transformNote(*note)
            #inds[instr_to_play].append(note)    
            current_clock().fork(piano_play_note,(pitch,volume,duration,isPause,0))        
        if len(zf)>0:
            note = zf.pop(0)
            pitch,duration,volume,isPause = transformNote(*note)
            #inds[instr_to_play].append(note)    
            current_clock().fork(piano_play_note,(pitch,volume,duration,isPause,0))                    
        if len(current_clock().children()) > 0:
            current_clock().wait_for_children_to_finish()
        #    #pass
        else:
            # prevents hanging if nothing has been forked yet
            current_clock().wait(1.0)    

dynamics = ["ppp","pp","p","mp","mf","f","ff","fff"]
vols = [(128//8)*(k+1) for k in range(8)]
ddvol = dict(zip(dynamics,vols)) 
            
def transformNote(pitch,durNum,durDenom,dyns,is_rest):
    global ddvol
    vol = ddvol[dyns]
    
    return pitch, durNum/durDenom*1.0*4.0,vol,is_rest=="is_rest"
            
def main():
    global s, s_forked,midiFileName,instrumentMidiNumber,knn_is_reverse,knn_nr_neighbors,inds,instr_to_play,loopKnn
    global d,zf,mf,rf,kf
    # https://stackoverflow.com/questions/24072790/how-to-detect-key-presses
    #try:  # used try so that if user pressed other than the given key error will not be shown
    #    if keyboard.is_pressed('q'):  # if key 'q' is pressed 
    #        print('You Pressed A Key!')
    #        break  # finishing the loop
    #except:
    #    break 
        
    while True:
        if not s_forked:
            s_forked = True
            s.fork(scamp_loop)
        imageInLoop()

            
        #if not zeigefinger_note is None:
        #    append_knn_notes(transformNote(*zeigefinger_note))
        #if not mittelfinger_note is None:
        #    append_knn_notes(transformNote(*mittelfinger_note))
    
    
    cap.release()
    cv2.destroyAllWindows()


instr_to_play = 0            
inds = []
global_counter_midi_pressed = 0            
started_transcribing = False
s_forked = False
knn_notes_to_play_for_marimba = []        
       
pressed_notes = []
started_notes = []


# soundfonts from https://sites.google.com/site/soundfonts4u/home
generalSF = "/usr/share/sounds/sf3/MuseScore_General.sf3"
pianoSF = "~/Dokumente/MuseScore3/SoundFonts/4U-Mellow-Steinway-v3.6.sf2"
    
if len(sys.argv) == 2:
    conf = readConfiguration(sys.argv[1])
else:
    conf = {
        "soundfont": pianoSF, 
        "loopKnn" : False,
        "knn_nr_neighbors" : 2,
        "knn_is_reverse" : True,
        "instrumentName" : "Mellow Steinway",
        "midiFileName" : "./midi/live_music_grand_piano.mid",
        "knn_model" : "./knn_models/knn.pkl",
        "jump" : False # will jump directly to the k-th nearest neighbor, without looking at the other nearest neighbors.
    }
    writeConfiguration("./scamp-start-conf.yaml",conf)

loopKnn = conf["loopKnn"]    
knn_nr_neighbors = conf["knn_nr_neighbors"]
knn_is_reverse=conf["knn_is_reverse"]    
instrumentName = conf["instrumentName"]    
midiFileName = conf["midiFileName"]
instrumentMidiNumber = get_general_midi_number_for_instrument_name(instrumentName,generalSF)    
std = [instrumentName]
sf = conf["soundfont"]
list_nbrs_notes_octaves = load_knn(conf["knn_model"])
jump = conf["jump"]
    
tracks = construct_ensemble(sf,std)

print(tracks)
print(len(tracks))

tempo = 80
s = Session(tempo=tempo,default_soundfont=sf).run_as_server()

s.print_available_midi_output_devices()
#print(dir(s))

s.print_available_midi_input_devices()

    
for t in tracks:
    print(t[1][0].preset)
    s.add_instrument(t[0])

#piano = s.new_part("Mellow Steinway")
#harp = s.new_part("Mellow Steinway")
#marimba = s.new_part("Marimba")

instrs = s.instruments #[piano,harp]

knn_notes_to_play = [] 

for i in instrs:
    inds.append([])
    knn_notes_to_play.append([])

#s.add_instrument(harp)
#s.add_instrument(piano)
#s.add_instrument(marimba)

   
#s.register_midi_listener(port_number_or_device_name="LPK25", callback_function=callback_midi)

#s.start_transcribing()


#s.wait_forever()
#s.run_as_server()



    
    
if __name__=="__main__":
    main()