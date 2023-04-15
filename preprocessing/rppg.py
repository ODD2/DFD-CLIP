import os 
import math
import pickle
import heartpy as hp
from src.datasets import resample
from matplotlib import pyplot as plt


def measure(_hr_data,sample_rate,visual=False):
    # - preprocess the ERG data: filter out the noise.
    _hr_data = hp.filter_signal(_hr_data, cutoff = 0.05, sample_rate = sample_rate, filtertype='notch')
    # - scale down the ERG value to 3.4 max.
    _hr_data = (_hr_data - _hr_data.min())/(_hr_data.max()-_hr_data.min()) * 3.4
    # - resample the ERG
    _hr_data = resample(_hr_data, len(_hr_data) * 10)
    # - process the ERG data: get measurements.
    _wd, _measures = hp.process(hp.scale_data(_hr_data),sample_rate * 10)

    # - simple validations on the measures
    for v in _measures.values():
        # ignore
        if type(v)==float and math.isnan(v):
            raise Exception("rPPG signale measure error.")

    if visual:
        visualize(_hr_data, _wd,_measures)

    return _wd, _measures


def visualize(hrdata,wd,measures):
    #display computed results and measures
    plt.figure(figsize=(6.4,4.8))
    plt.plot(list(range(len(hrdata))),hrdata)
    plt.ion()
    plt.show()
    hp.plotter(wd, measures)
    for measure in measures.keys():
        print('%s: %f' %(measure, measures[measure]))



def pre_calculate_bpm(segment_duration=10,signal_offset=30):
    from tqdm import tqdm
    from glob import glob
    from pyedflib import highlevel as reader

    for bdf_file in tqdm(glob("./datasets/hci/Sessions/*/*.bdf")):
        # heart rate data processing
        signals, signal_headers, _ = reader.read_edf(bdf_file,ch_names=["EXG1","EXG2","EXG3","Status"])

        assert len(set([ int(i["sample_frequency"]) for i in  signal_headers])) == 1
        
        sample_rate = signal_headers[0]["sample_frequency"]
        begin_sample = int(sample_rate * signal_offset)
        end_sample = int(len(signals[0]) - (sample_rate * signal_offset))
        total_samples = end_sample - begin_sample

        assert total_samples > 0

        total_segments = int((total_samples/sample_rate)//segment_duration)

        if(not total_segments > 0):
            continue


        segment_sample = []
        segment_measure = []

        try:
            for segment in range(total_segments):
                _hr_datas = []
                hr_segment_offset = int(begin_sample + segment * segment_duration * sample_rate)
                # - the amount of samples for the duration of a clip
                hr_segment_samples = int(segment_duration*sample_rate)

                if(segment == total_segments - 1):
                    hr_segment_end = end_sample
                else:
                    hr_segment_end = hr_segment_offset + hr_segment_samples

                for hr_channel_idx in range(3):
                    try:
                        # - fetch heart rate data of clip duration
                        _hr_data = signals[hr_channel_idx][hr_segment_offset:hr_segment_end]
                        _wd,_measures =  measure(_hr_data,sample_rate,False)
                        if(_measures["bpm"] <41 or _measures["bpm"]>180):
                            continue
                        # - save for comparison.
                        _hr_datas.append((_hr_data,_measures,_wd))
                    except Exception as e:
                        print(f"Error occur during heart rate analysis for channel {hr_channel_idx}:{e}")
                        continue

                if(len(_hr_datas) == 0):
                    raise Exception(f"Unable to process the ERG data for segment {segment}")
 

                # get the best ERG measurement result with the sdnn
                best_pair = sorted(_hr_datas,key=lambda x : x[1]["sdnn"])[0]
                hr_data,measures,wd = best_pair[0], best_pair[1], best_pair[2]
                segment_sample.append(hr_segment_end)
                segment_measure.append(measures)
                # visualize(hr_data,wd,measures)
                # plt.waitforbuttonpress()
                # plt.close("all")
        except Exception as e:
            print(f"Error Occur for file:{bdf_file}({e})")
            continue

        if(len(segment_measure) == 0):
            continue

        segment_measure.insert(0,segment_measure[0])
        segment_sample.insert(0,begin_sample)
        session_folder =  '/'.join(bdf_file.split('/')[:-1])
        measure_folder = session_folder.replace("Sessions","Measures")
        os.makedirs(measure_folder,exist_ok=True)
        with open(os.path.join(measure_folder,"data.pickle"),"wb") as f:
            pickle.dump({"idx": segment_sample, "data": segment_measure},f)
            
            


            



if __name__ == "__main__":
    pre_calculate_bpm()