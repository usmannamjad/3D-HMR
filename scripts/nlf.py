import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
import pickle
import gc
import time
import warnings
warnings.filterwarnings("ignore")


MODEL_PATH = '/home/usmannamjad/vh/models/nlf_l_multi.torchscript'
VIDEOS_PATH = Path('/home/usmannamjad/vh/datasets/Motion-X++/video/')
OUTPUT_PATH = Path('/home/usmannamjad/vh/datasets/outputs/nlf')
DATASET_SUBSETS = ['music', 'animation', 'kungfu', 'perform']
REMAINING_VIDEOS = True

# Load the TorchScript model
model = torch.jit.load(MODEL_PATH).cuda().eval()
batch_size = 128
key_names = ['boxes', 'pose', 'betas', 'trans', 'vertices3d', 
             'joints3d', 'vertices2d', 'joints2d', 'vertices3d_nonparam', 
             'joints3d_nonparam', 'vertices2d_nonparam', 'joints2d_nonparam', 
             'vertex_uncertainties', 'joint_uncertainties']

def get_total_paths():
    paths = []
    for video_path in get_video_paths():
        paths.append(video_path)
    total_paths = len(paths)
    del paths
    return total_paths


def get_video_paths():
    for subset in DATASET_SUBSETS:
        subset_path = VIDEOS_PATH / subset
        if subset_path.exists() and subset_path.is_dir():
            for video_file in subset_path.rglob("*.mp4"):  # Recursively finds .mp4 files
                yield video_file  # Yields full path as a generator



videos_not_processed = []

# if REMAINING_VIDEOS:
#     with open(OUTPUT_PATH / "videos_not_processed.txt", 'r') as f:
#         paths = [path.strip() for path in f]
#     total_paths = len(paths)
# else:
#     paths = get_video_paths()   
#     total_paths = get_total_paths()    



# paths = ['Play_Banhu_33_clip2', 'Play_Cello_7_clip1', 'Play_Snare_drum_15_clip2', 'Ancient_Drum_4_clip1', 'beat_drums_and_gongs_56_clip1', 'Play_the_stringed_guqin_60_clip1', 'Play_Gaohu_7_clip3', 'Play_the_violin_60_clip1_clip3', 'Play_the_panpipe_18_clip1', 'Ancient_Drum_64_clip1', 'Play_Cymbals_11_clip1', 'Play_the_stringed_guqin_50_clip2', 'Play_Dulcimer_42_clip1', 'Play_pipa_54_clip2', 'Play_accordion_11_clip1', 'beat_drums_and_gongs_32_clip1', 'Play_Viola_21_clip8', 'Beat_Chime_17_clip2', 'waist_drum_7_clip1_clip1', 'beat_drums_and_gongs_48_clip5', 'Play_Erhu_22_clip1', 'Ancient_Drum_9_clip5', 'Play_Bassoon_1_clip1', 'Play_Carillon_12_clip1', 'Play_Banhu_30_clip3', 'Ancient_Drum_3_clip6', 'beat_drums_and_gongs_29_clip1', 'Ancient_Drum_56_clip1', 'Play_Viola_21_clip1', 'Ancient_Drum_10_clip6', 'Play_pipa_55_clip2', 'Play_Clarinet_25_clip2', 'Play_the_violin_61_clip1', 'Play_Xun_19_clip1', 'Play_pipa_2_clip1_clip1', 'Play_Bass_25_clip2', 'Ancient_Drum_15_clip8', 'Play_Zither_1_clip1', 'Play_pipa_29_clip1', 'Play_the_stringed_guqin_55_clip2', 'beat_drums_and_gongs_42_clip3', 'Play_Cello_16_clip1', 'Beat_Chime_17_clip6', 'Play_Cymbals_15_clip1', 'Play_Flute_30_clip1', 'Play_Ruan_60_clip2', 'Play_pipa_1_clip1', 'Play_Guitar_14_clip1', 'Play_the_stringed_guqin_44_clip1', 'Play_Viola_9_clip1', 'Play_Dulcimer_3_clip1', 'Play_Ukulele_15_clip1', 'Play_Guitar_10_clip1', 'Play_the_violin_12_clip3', 'Play_pipa_40_clip1', 'Play_pipa_58_clip1', 'Play_Dulcimer_19_clip1', 'Play_Clarinet_5_clip1', 'Play_Guitar_25_clip2', 'Play_the_violin_46_clip1', 'Play_accordion_18_clip1', 'Play_the_stringed_guqin_39_clip2', 'Play_Snare_drum_18_clip1', 'waist_drum_2_clip2', 'Play_Sax_46_clip2', 'Play_French_Horn_23_clip1', 'Play_the_stringed_guqin_45_clip1', 'play_the_flute_21_clip1', 'Ancient_Drum_3_clip2', 'Play_the_stringed_guqin_23_clip1', 'Play_Sheng_2_clip2', 'Play_the_violin_62_clip1', 'Play_Zither_30_clip2', 'Play_Cello_41_clip1', 'Beat_Chime_22_clip1', 'Ancient_Drum_16_clip1', 'Play_the_violin_38_clip6', 'Play_pipa_6_clip1', 'Play_Cello_6_clip1', 'Play_Flute_26_clip2', 'Play_the_stringed_guqin_53_clip1', 'beat_drums_and_gongs_32_clip2', 'Play_horse_head_string_instrument_3_clip1', 'Play_Threestringed_plucked_instrument_30_clip1', 'Ancient_Drum_41_clip1', 'Play_Cello_42_clip1', 'Play_Timpani_14_clip1', 'Ancient_Drum_32_clip1', 'Play_Cello_29_clip1', 'Play_Drum_Set_13_clip1', 'Play_Harp_20_clip1', 'Play_Bass_2_clip4', 'Play_Drum_Set_19_clip2', 'Play_the_violin_60_clip1_clip2', 'Play_Cello_31_clip1', 'Play_pipa_16_clip1', 'Play_Malimba_xylophone_34_clip2', 'Play_Malimba_xylophone_27_clip1', 'Play_Zither_23_clip1', 'Play_Dulcimer_17_clip1', 'Play_the_stringed_guqin_58_clip1', 'Play_bass_drum_9_clip1', 'Play_the_stringed_guqin_40_clip2', 'Play_the_stringed_guqin_51_clip1', 'Play_horse_head_string_instrument_5_clip1', 'Play_pipa_16_clip2', 'Play_Banhu_18_clip2', 'Play_pipa_17_clip3', 'Play_Trombone_11_clip3', 'Play_Sheng_34_clip1', 'Play_Gongs_and_drums_2_clip1', 'Ancient_Drum_14_clip5', 'Play_horse_head_string_instrument_7_clip2', 'Play_accordion_6_clip1', 'Play_French_Horn_11_clip1', 'Play_Malimba_xylophone_29_clip1', 'Play_Malimba_xylophone_23_clip2', 'Play_Bass_11_clip1', 'Play_Malimba_xylophone_33_clip1', 'Play_Trumpet_33_clip1_clip2', 'Ancient_Drum_16_clip7', 'Play_Piccolo_1_clip1', 'Play_Tuba_24_clip2', 'Play_Viola_28_clip4', 'beat_drums_and_gongs_22_clip1', 'Play_the_stringed_guqin_48_clip1', 'Ancient_Drum_10_clip4', 'Play_pipa_39_clip1', 'Play_the_violin_41_clip7', 'Ancient_Drum_64_clip2', 'Play_Harp_26_clip1', 'Ancient_Drum_10_clip7', 'Play_the_stringed_guqin_53_clip2', 'Play_Threestringed_plucked_instrument_42_clip1', 'Play_Viola_7_clip5', 'Ancient_Drum_16_clip5', 'Play_the_piano_9_clip1', 'Beat_Chime_17_clip7', 'Play_Timpani_11_clip1', 'Play_French_Horn_22_clip1', 'Ancient_Drum_16_clip8', 'Ancient_Drum_15_clip2', 'Play_the_stringed_guqin_9_clip1', 'Play_Suona_10_clip1_clip1', 'Play_the_violin_26_clip1', 'Play_Bass_3_clip3', 'Ancient_Drum_13_clip6', 'Play_the_violin_21_clip1', 'Play_Doublet_3_clip1', 'Play_Ukulele_25_clip1', 'Ancient_Drum_12_clip2', 'beat_drums_and_gongs_47_clip3', 'Play_Bass_55_clip1', 'Play_Ruan_34_clip1', 'Play_bass_drum_4_clip1_clip2', 'Ancient_Drum_54_clip2', 'Play_Timpani_11_clip2', 'Play_Timpani_9_clip1_clip1', 'Play_Timpani_5_clip1', 'Play_Sax_34_clip1', 'Ancient_Drum_11_clip7', 'Play_Cello_8_clip1', 'Play_Cello_12_clip1', 'Play_Xun_11_clip2', 'Play_Sax_48_clip1', 'Play_Dulcimer_20_clip1', 'Play_the_piano_36_clip2', 'Play_Viola_21_clip5', 'Play_Timpani_9_clip1', 'Play_Ukulele_25_clip2', 'Play_Harp_31_clip1', 'Play_the_stringed_guqin_54_clip1', 'Play_Dulcimer_44_clip2', 'Play_Sax_47_clip1', 'Play_Guitar_39_clip1', 'Play_Banhu_6_clip1', 'Play_Ruan_32_clip1', 'Play_Sheng_37_clip1', 'Play_Malimba_xylophone_26_clip2', 'waist_drum_6_clip1', 'Play_Trumpet_2_clip1_clip2', 'Play_Ukulele_36_clip1', 'Play_Timpani_39_clip3', 'Play_Xun_11_clip1', 'Play_Xun_18_clip1', 'Play_Bass_28_clip9', 'Play_Tuba_24_clip1', 'beat_drums_and_gongs_44_clip5', 'Ancient_Drum_1_clip3', 'Play_Trumpet_27_clip1', 'Play_Dulcimer_7_clip1', 'Play_pipa_18_clip1', 'Play_the_stringed_guqin_55_clip1', 'Play_Xun_22_clip1', 'Play_pipa_25_clip1', 'play_the_flute_17_clip1', 'Play_Sheng_2_clip1', 'beat_drums_and_gongs_7_clip1', 'Play_Big_Ruan_9_clip1', 'Ancient_Drum_13_clip1', 'play_the_flute_7_clip1', 'Ancient_Drum_8_clip4', 'Ancient_Drum_8_clip6', 'Ancient_Drum_9_clip11', 'Play_Cello_35_clip1', 'Play_Gongs_and_drums_19_clip1', 'Play_Suona_55_clip1', 'Play_Zither_26_clip2', 'Play_Guitar_11_clip2', 'Play_Sheng_25_clip1', 'Play_the_stringed_guqin_41_clip2', 'Play_Bass_14_clip1', 'Play_Traditional_Chinese_drum_3_clip1', 'Play_Threestringed_plucked_instrument_44_clip4', 'Play_bass_drum_7_clip1', 'beat_drums_and_gongs_1_clip1', 'Play_Xun_20_clip2', 'Play_Cello_47_clip1', 'play_the_flute_30_clip1', 'Play_French_Horn_13_clip2', 'beat_drums_and_gongs_11_clip1', 'Baduanjin_7_clip2', 'Play_Malimba_xylophone_22_clip2', 'Play_Banhu_29_clip1', 'Play_Threestringed_plucked_instrument_21_clip1', 'waist_drum_8_clip1', 'Play_French_Horn_17_clip1', 'Play_Trumpet_23_clip5', 'Play_Jinghu_28_clip1', 'Play_Carillon_5_clip1', 'Beat_Chime_12_clip1', 'Ancient_Drum_3_clip5', 'Beat_Chime_17_clip1', 'Play_Dulcimer_18_clip2', 'Play_pipa_26_clip1', 'Play_Banhu_2_clip2', 'Play_the_piano_29_clip1', 'Play_Xun_48_clip1', 'Play_Bass_1_clip1', 'Play_Harp_25_clip1', 'Beat_Chime_18_clip1', 'Play_Banhu_32_clip2', 'Play_Bass_29_clip3', 'Play_Viola_26_clip4', 'Play_the_violin_36_clip2', 'Play_Zither_4_clip1', 'Play_pipa_38_clip1', 'Play_the_piano_32_clip1', 'Play_the_violin_61_clip1_clip2', 'Play_Malimba_xylophone_36_clip1', 'Play_Ukulele_25_clip3', 'Play_Oboe_26_clip1', 'Play_the_violin_44_clip1_clip1', 'Play_pipa_19_clip1', 'Play_the_stringed_guqin_13_clip1', 'Play_accordion_1_clip1', 'Play_Gaohu_7_clip2', 'Play_Threestringed_plucked_instrument_12_clip1', 'Play_Viola_11_clip1', 'Play_Trumpet_31_clip1', 'Play_Ukulele_17_clip1', 'Ancient_Drum_59_clip2', 'Play_Big_Ruan_8_clip1_clip1', 'Play_Cello_22_clip1', 'Ancient_Drum_37_clip3', 'Play_Viola_20_clip4', 'Play_the_violin_55_clip2', 'Play_the_violin_24_clip2', 'Play_Malimba_xylophone_28_clip1', 'Play_Threestringed_plucked_instrument_9_clip2', 'Play_pipa_15_clip1', 'Play_Banhu_3_clip1', 'Play_the_stringed_guqin_43_clip1', 'Play_Sheng_5_clip1', 'Play_the_violin_52_clip1', 'Play_Dulcimer_14_clip1_clip3', 'play_the_flute_19_clip1', 'Play_Viola_14_clip6', 'Play_Bass_28_clip7', 'Ancient_Drum_58_clip1', 'Play_Bass_25_clip5', 'Play_Bass_5_clip2', 'play_the_flute_16_clip1', 'Play_pipa_48_clip1', 'Play_accordion_26_clip1', 'Play_Suona_30_clip1', 'Play_Ukulele_20_clip1_clip1', 'Play_Gongs_and_drums_9_clip1', 'Play_Guitar_30_clip1', 'Play_Tuba_28_clip2', 'play_Electric_guitar_43_clip1', 'Play_Jinghu_29_clip2', 'Play_the_violin_5_clip2', 'Play_Sax_49_clip2', 'Play_Viola_10_clip1', 'Play_Dulcimer_43_clip1', 'Play_Dulcimer_31_clip1', 'Play_the_piano_14_clip2', 'Play_Gongs_and_drums_1_clip1', 'Play_Threestringed_plucked_instrument_54_clip1', 'Play_Harp_40_clip1', 'Play_Viola_8_clip1', 'Play_Ruan_62_clip1', 'Play_accordion_10_clip1', 'Play_Drum_Set_12_clip1_clip1', 'Play_Traditional_Chinese_drum_2_clip1', 'Ancient_Drum_7_clip7', 'Play_Sheng_29_clip1', 'Play_Ruan_44_clip1', 'Play_the_stringed_guqin_47_clip1', 'Play_the_violin_16_clip1_clip2', 'Play_Bass_23_clip1', 'Play_French_Horn_26_clip1', 'Play_pipa_54_clip1', 'beat_drums_and_gongs_44_clip1', 'Play_Carillon_8_clip1_clip3', 'Play_the_stringed_guqin_71_clip1', 'Play_Threestringed_plucked_instrument_20_clip1', 'Play_Malimba_xylophone_7_clip1', 'beat_drums_and_gongs_30_clip2', 'Ancient_Drum_29_clip2', 'Play_Erhu_8_clip1', 'Play_Cello_27_clip1', 'Play_Tuba_27_clip1', 'Play_the_piano_4_clip2', 'Play_Sax_45_clip1', 'Play_Erhu_18_clip1', 'Play_Cello_20_clip2', 'Play_Gaohu_27_clip1', 'Play_Guitar_11_clip1', 'Play_Trombone_18_clip1', 'Play_Flute_25_clip2', 'Play_Gaohu_5_clip2', 'Play_Doublet_1_clip3', 'Play_Ruan_33_clip1', 'Play_Timpani_29_clip1', 'Play_Bass_12_clip5', 'Play_Suona_17_clip1', 'Play_Bass_25_clip4', 'Play_the_violin_61_clip1_clip1', 'Play_Timpani_32_clip1', 'Play_Malimba_xylophone_19_clip1', 'beat_drums_and_gongs_26_clip2', 'play_Electric_guitar_49_clip2', 'Play_horse_head_string_instrument_11_clip1', 'play_the_veritical_flute_7_clip2', 'Ancient_Drum_12_clip5', 'Play_accordion_21_clip2', 'Ancient_Drum_8_clip7', 'Play_Ukulele_5_clip1', 'Play_Tuba_6_clip1', 'Play_pipa_10_clip2', 'play_the_flute_18_clip1', 'Play_Piccolo_5_clip1']
# paths = [f"/home/usmannamjad/vh/datasets/Motion-X++/video/music/{p}.mp4" for p in paths]
# paths = ["/home/usmannamjad/vh/gangnam.mp4"]
paths = ["/home/usmannamjad/vh/tai_chi.mp4"]
total_paths = len(paths)
for video_path in tqdm(paths, total=total_paths):
    try:
        frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")

        # Convert frames to tensor and move to GPU
        frames = frames.permute(0, 3, 1, 2).cuda()  # Shape: (num_frames, C, H, W)

        # Process video frames in batches
        num_frames = frames.shape[0]
        print("frames shape: ", frames.shape, num_frames)

        results = {key: [] for key in key_names}
        # with torch.inference_mode():
        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                print("i: ", i)
                print("i + batch_size: ", i + batch_size)
                frame_batch = frames[i:i+batch_size]
                preds = model.detect_smpl_batched(frame_batch, model_name='smplx')

                for key in preds:
                    results[key].extend([p.cpu() for p in preds[key]])
    

        del frames
        torch.cuda.empty_cache()
        gc.collect()
        print("output: ", len(results['pose']))
        print("output: ", len(results['betas']))
        video_path = str(video_path).replace('.mp4', '')
        video_path = video_path.split('/')
        video_name = video_path[-1]
        subset = video_path[-2]
        save_path = OUTPUT_PATH / subset / f"{video_name}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        # del results
        time.sleep(1)

        print("reading picklefile")
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        print("results: ", len(results['pose']))
        print("results: ", len(results['betas']))
    except Exception as e:
        videos_not_processed.append(str(video_path))
        print(f"Error processing video: {video_path}")
        print(e)
        

print(f"Videos not processed:", len(videos_not_processed))
output_file = OUTPUT_PATH / "videos_not_processed.txt"
with open(output_file, 'w') as f:
    f.write("\n".join(videos_not_processed))
        
        