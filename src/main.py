import os
from pathlib import Path

from src.video_block_match import VideoBlockMatch

main_path = Path(__file__).parents[1]
video_path = os.path.join(main_path,"foreman-orig.avi")


orig_video_block_matcher = VideoBlockMatch(video_path=video_path,block_size=16,step_size=8)
orig_video_block_matcher.match_with_original()
orig_video_block_matcher.save_results("frames/from_orig")


pred_video_block_matcher = VideoBlockMatch(video_path=video_path,block_size=16,step_size=8)
pred_video_block_matcher.match_with_predicted()
pred_video_block_matcher.save_results("frames/from_pred")

