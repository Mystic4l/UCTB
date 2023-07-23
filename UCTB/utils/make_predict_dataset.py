def save_predict_in_tsv(data_loader, prediction, method, output_dir='../tsv/'):
    data_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'data')

    original_file = os.path.join(data_dir, "{}_{}.pkl".format(
        data_loader.dataset.dataset, data_loader.dataset.city))

    with open(original_file, "rb") as fp:
        pred_data = pickle.load(fp)
        pred_data['Pred'] = {}

    loader_id = data_loader.loader_id
    if loader_id not in pred_data["Pred"].keys():
        pred_data['Pred'][loader_id] = {}

    pred_data['Pred'][loader_id]["GroundTruth"] = np.squeeze(
        data_loader.test_y)

    if loader_id.endswith("N"):
        # use NodeTrafficLoader
        pred_data['Pred'][loader_id][method] = {}
        pred_data['Pred'][loader_id][method]["traffic_data_index"] = data_loader.traffic_data_index
        pred_data['Pred'][loader_id][method]["TrafficNode"] = np.squeeze(prediction)

    if loader_id.endswith("G"):
        # use GridTrafficLoader
        pred_data['Pred'][loader_id][method] = {}
        pred_data['Pred'][loader_id][method]["TrafficGrid"] = np.squeeze(prediction)

    # 获取数据集名称
    file_name_without_extension = data_loader.dataset.dataset + data_loader.dataset.city

    # 生成TSV文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_option = list(pred_data["Pred"].keys())[0]

    gt_list = pred_data["Pred"][dataset_option]["GroundTruth"]
    pd_list = pred_data["Pred"][dataset_option][method]["TrafficNode"]
    station_info = pred_data["Node"]["StationInfo"]

    gt_df = pd.DataFrame(gt_list)
    pd_df = pd.DataFrame(pd_list)
    gt_df = gt_df.transpose()
    pd_df = pd_df.transpose()

    station_info_df = pd.DataFrame(station_info)
    station_info_df = station_info_df.drop(station_info_df.columns[[0, 1, 4]], axis=1)

    try:
        gt_df.to_csv(output_dir + file_name_without_extension + '_gt.tsv', sep='\t', index=False, header=False)
        pd_df.to_csv(output_dir + file_name_without_extension + '_pd.tsv', sep='\t', index=False, header=False)
        station_info_df.to_csv(output_dir + file_name_without_extension + '_station_info.tsv', sep='\t', index=False,
                               header=False)
        print("TSV files generated successfully!")
    except Exception as e:
        print("Error while generating TSV files:", e)
        
    return pred_data['TimeRange'], pred_data['TimeFitness']
