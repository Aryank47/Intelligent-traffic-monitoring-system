from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou



def evaluate_model(model, dataloader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                pred_labels = output['labels'].cpu()

                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()

                # Filter by confidence threshold
                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                matched_gt = set()
                for pb, pl in zip(pred_boxes, pred_labels):
                    found_match = False
                    for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if i in matched_gt:
                            continue
                        iou = compute_iou(pb.numpy(), gb.numpy())
                        if iou >= iou_threshold and pl == gl:
                            all_preds.append(1)  # TP
                            all_gts.append(1)
                            matched_gt.add(i)
                            found_match = True
                            break
                    if not found_match:
                        all_preds.append(1)  # FP
                        all_gts.append(0)

                for i in range(len(gt_boxes)):
                    if i not in matched_gt:
                        all_preds.append(0)  # FN
                        all_gts.append(1)

    precision = precision_score(all_gts, all_preds)
    recall = recall_score(all_gts, all_preds)
    f1 = f1_score(all_gts, all_preds)
    accuracy = accuracy_score(all_gts, all_preds)

    print(f"📊 Precision: {precision:.4f}")
    print(f"📊 Recall:    {recall:.4f}")
    print(f"📊 F1 Score:  {f1:.4f}")
    print(f"📊 Accuracy:  {accuracy:.4f}")