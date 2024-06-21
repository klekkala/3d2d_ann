import open3d as o3d
import os
import pickle


def run_conversions_in_parallel(total_folder, pcls, legos):
    # a = ['2023_03_11',  '2023_03_29',  '2023_06_19',  '2023_06_21',  '2023_06_24',  '2023_06_28']
    for parent_folder in os.listdir(total_folder):
        print(parent_folder)
        # if parent_folder!= '2023_06_19':
        #     continue
        parent_folder = os.path.join(total_folder, parent_folder)
        if not os.path.isdir(parent_folder):
            continue
        for session in os.listdir(parent_folder):
            session_path = os.path.join(parent_folder, session)
            if not os.path.isdir(session_path):
                continue
            pcl_path = os.path.join(session_path, 'all_pcl')
            lego_path = os.path.join(session_path, 'all_lego')
            print(len(os.listdir(pcl_path)))
            for pcd in os.listdir(pcl_path)[:5]:

                pcls.append(len(o3d.io.read_point_cloud(os.path.join(pcl_path, pcd)).points))
            legos.append(len(o3d.io.read_point_cloud(os.path.join(lego_path, 'surfaceMap.pcd')).points))
                
    return pcls, legos



pcls = []
legos = []

pcls, legos = run_conversions_in_parallel('/lab/tmpig13b/kiran/bag_dump', pcls, legos)
pcls, legos = run_conversions_in_parallel('/lab/tmpig13b/kiran/error_dump', pcls, legos)

with open('hist.pkl', 'wb') as f:
    pickle.dump((pcls, legos), f)