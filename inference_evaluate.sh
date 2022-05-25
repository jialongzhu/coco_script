MODEL_NAME="faster_rcnn"
DATASET_TYPE="grass"
EXP_NAME="faster_rcnn_mobilenetv2_grass"
LOG_JSON_FILE="20220522_002829.log.json"
RESULT_PKL="result.pkl"
INPUT_SHAPE="512 512"

# 评估mAP
python tools/test.py configs/${MODEL_NAME}/${EXP_NAME}.py work_dirs/${EXP_NAME}/epoch_27.pth --eval bbox proposal --out work_dirs/${EXP_NAME}/result.pkl --show-score-thr 0.5 --show-dir work_dirs/${EXP_NAME}/infered_images --eval-options "classwise=True">work_dirs/${EXP_NAME}/mAP.txt
# 分类损失和bbox损失图像
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/${EXP_NAME}/${LOG_JSON_FILE} --key loss_cls loss_bbox --legend loss_cls loss_bbox --out work_dirs/${EXP_NAME}/loss.pdf
# 绘制准确率曲线
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/${EXP_NAME}/${LOG_JSON_FILE} --keys acc --legend acc --out work_dirs/${EXP_NAME}/acc.pdf
# 训练时间
python tools/analysis_tools/analyze_logs.py cal_train_time work_dirs/${EXP_NAME}/${LOG_JSON_FILE}>work_dirs/${EXP_NAME}/train_time.txt
# 评估params, flops
python tools/analysis_tools/get_flops.py configs/${MODEL_NAME}/${EXP_NAME}.py --shape ${INPUT_SHAPE}>work_dirs/${EXP_NAME}/flops.txt
# 混淆矩阵
python tools/analysis_tools/confusion_matrix.py configs/${MODEL_NAME}/${EXP_NAME}.py work_dirs/${EXP_NAME}/${RESULT_PKL} work_dirs/${EXP_NAME}
# 评估FPS
python ./tools/analysis_tools/benchmark.py configs/${MODEL_NAME}/${EXP_NAME}.py work_dirs/${EXP_NAME}/epoch_27.pth --fuse-conv-bn>work_dirs/${EXP_NAME}/fps.txt
