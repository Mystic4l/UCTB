Search.setIndex({docnames:["APIReference","UCTB","UCTB.dataset","UCTB.evaluation","UCTB.model","UCTB.model_unit","UCTB.preprocess","UCTB.train","UCTB.utils","index","md_file/all_results","md_file/index","md_file/installation","md_file/introduction","md_file/quickstart","md_file/src/image/README","md_file/static/current_supported_models","md_file/static/experiment_on_bike","md_file/static/experiment_on_chargestation","md_file/static/experiment_on_didi","md_file/static/experiment_on_metro","md_file/static/parameter_search","md_file/static/preprocess_api","md_file/static/quick_start","md_file/static/stable_test","md_file/static/stmeta","md_file/static/transfer_record","md_file/tutorial","md_file/uctb_group","modules","update_guide"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,sphinx:56},filenames:["APIReference.rst","UCTB.rst","UCTB.dataset.rst","UCTB.evaluation.rst","UCTB.model.rst","UCTB.model_unit.rst","UCTB.preprocess.rst","UCTB.train.rst","UCTB.utils.rst","index.rst","md_file\\all_results.md","md_file\\index.md","md_file\\installation.md","md_file\\introduction.md","md_file\\quickstart.md","md_file\\src\\image\\README.md","md_file\\static\\current_supported_models.md","md_file\\static\\experiment_on_bike.md","md_file\\static\\experiment_on_chargestation.md","md_file\\static\\experiment_on_didi.md","md_file\\static\\experiment_on_metro.md","md_file\\static\\parameter_search.md","md_file\\static\\preprocess_api.md","md_file\\static\\quick_start.md","md_file\\static\\stable_test.md","md_file\\static\\stmeta.md","md_file\\static\\transfer_record.md","md_file\\tutorial.md","md_file\\uctb_group.md","modules.rst","update_guide.txt"],objects:{"":{UCTB:[1,0,0,"-"]},"UCTB.dataset":{data_loader:[2,0,0,"-"],dataset:[2,0,0,"-"]},"UCTB.dataset.data_loader":{GridTrafficLoader:[2,1,1,""],NodeTrafficLoader:[2,1,1,""],TransferDataLoader:[2,1,1,""]},"UCTB.dataset.data_loader.NodeTrafficLoader":{LM:[2,2,1,""],build_graph:[2,3,1,""],daily_slots:[2,2,1,""],dataset:[2,2,1,""],external_dim:[2,2,1,""],make_concat:[2,3,1,""],st_map:[2,3,1,""],station_number:[2,2,1,""],train_closeness:[2,2,1,""],train_y:[2,2,1,""]},"UCTB.dataset.data_loader.TransferDataLoader":{checkin_sim:[2,3,1,""],checkin_sim_sd:[2,3,1,""],poi_sim:[2,3,1,""],traffic_sim:[2,3,1,""],traffic_sim_fake:[2,3,1,""]},"UCTB.dataset.dataset":{DataSet:[2,1,1,""]},"UCTB.dataset.dataset.DataSet":{data:[2,2,1,""],node_monthly_interaction:[2,2,1,""],node_station_info:[2,2,1,""],node_traffic:[2,2,1,""],time_fitness:[2,2,1,""],time_range:[2,2,1,""]},"UCTB.evaluation":{metric:[3,0,0,"-"]},"UCTB.evaluation.metric":{mape:[3,4,1,""],mape_grid:[3,4,1,""],rmse:[3,4,1,""],rmse_grid:[3,4,1,""]},"UCTB.model":{ARIMA:[4,0,0,"-"],DCRNN:[4,0,0,"-"],DeepST:[4,0,0,"-"],GeoMAN:[4,0,0,"-"],HM:[4,0,0,"-"],HMM:[4,0,0,"-"],STMeta:[4,0,0,"-"],ST_MGCN:[4,0,0,"-"],ST_ResNet:[4,0,0,"-"],XGBoost:[4,0,0,"-"]},"UCTB.model.ARIMA":{ARIMA:[4,1,1,""]},"UCTB.model.ARIMA.ARIMA":{adf_test:[4,3,1,""],get_order:[4,3,1,""],predict:[4,3,1,""]},"UCTB.model.DCRNN":{DCRNN:[4,1,1,""]},"UCTB.model.DCRNN.DCRNN":{build:[4,3,1,""]},"UCTB.model.DeepST":{DeepST:[4,1,1,""]},"UCTB.model.DeepST.DeepST":{build:[4,3,1,""]},"UCTB.model.GeoMAN":{GeoMAN:[4,1,1,""],input_transform:[4,4,1,""],split_timesteps:[4,4,1,""]},"UCTB.model.GeoMAN.GeoMAN":{build:[4,3,1,""]},"UCTB.model.HM":{HM:[4,1,1,""]},"UCTB.model.HM.HM":{predict:[4,3,1,""]},"UCTB.model.HMM":{HMM:[4,1,1,""]},"UCTB.model.HMM.HMM":{fit:[4,3,1,""],predict:[4,3,1,""]},"UCTB.model.STMeta":{STMeta:[4,1,1,""]},"UCTB.model.STMeta.STMeta":{build:[4,3,1,""]},"UCTB.model.ST_MGCN":{ST_MGCN:[4,1,1,""]},"UCTB.model.ST_MGCN.ST_MGCN":{build:[4,3,1,""]},"UCTB.model.ST_ResNet":{ST_ResNet:[4,1,1,""]},"UCTB.model.ST_ResNet.ST_ResNet":{build:[4,3,1,""]},"UCTB.model.XGBoost":{XGBoost:[4,1,1,""]},"UCTB.model.XGBoost.XGBoost":{fit:[4,3,1,""],predict:[4,3,1,""]},"UCTB.model_unit":{BaseModel:[5,0,0,"-"],DCRNN_CELL:[5,0,0,"-"],GraphModelLayers:[5,0,0,"-"],ST_RNN:[5,0,0,"-"]},"UCTB.model_unit.BaseModel":{BaseModel:[5,1,1,""]},"UCTB.model_unit.BaseModel.BaseModel":{add_summary:[5,3,1,""],build:[5,3,1,""],close:[5,3,1,""],fit:[5,3,1,""],load:[5,3,1,""],load_event_scalar:[5,3,1,""],manual_summary:[5,3,1,""],predict:[5,3,1,""],save:[5,3,1,""]},"UCTB.model_unit.DCRNN_CELL":{DCGRUCell:[5,1,1,""]},"UCTB.model_unit.DCRNN_CELL.DCGRUCell":{call:[5,3,1,""],compute_output_shape:[5,3,1,""],output_size:[5,3,1,""],state_size:[5,3,1,""]},"UCTB.model_unit.GraphModelLayers":{GAL:[5,1,1,""],GCL:[5,1,1,""],GraphBuilder:[5,1,1,""]},"UCTB.model_unit.GraphModelLayers.GAL":{add_ga_layer_matrix:[5,3,1,""],add_residual_ga_layer:[5,3,1,""],attention_merge_weight:[5,3,1,""]},"UCTB.model_unit.GraphModelLayers.GCL":{add_gc_layer:[5,3,1,""],add_multi_gc_layers:[5,3,1,""]},"UCTB.model_unit.GraphModelLayers.GraphBuilder":{adjacent_to_laplacian:[5,3,1,""],correlation_adjacent:[5,3,1,""],distance_adjacent:[5,3,1,""],haversine:[5,3,1,""],interaction_adjacent:[5,3,1,""]},"UCTB.model_unit.ST_RNN":{GCLSTMCell:[5,1,1,""]},"UCTB.model_unit.ST_RNN.GCLSTMCell":{build:[5,3,1,""],call:[5,3,1,""],kth_cheby_ploy:[5,3,1,""]},"UCTB.preprocess":{preprocessor:[6,0,0,"-"],time_utils:[6,0,0,"-"]},"UCTB.preprocess.preprocessor":{MoveSample:[6,1,1,""],Normalizer:[6,1,1,""],ST_MoveSample:[6,1,1,""],SplitData:[6,1,1,""]},"UCTB.preprocess.preprocessor.MoveSample":{general_move_sample:[6,3,1,""]},"UCTB.preprocess.preprocessor.Normalizer":{min_max_denormal:[6,3,1,""],min_max_normal:[6,3,1,""]},"UCTB.preprocess.preprocessor.ST_MoveSample":{move_sample:[6,3,1,""]},"UCTB.preprocess.preprocessor.SplitData":{split_data:[6,3,1,""],split_feed_dict:[6,3,1,""]},"UCTB.preprocess.time_utils":{is_valid_date:[6,4,1,""],is_work_day_america:[6,4,1,""],is_work_day_china:[6,4,1,""]},"UCTB.train":{EarlyStopping:[7,0,0,"-"],MiniBatchTrain:[7,0,0,"-"]},"UCTB.train.EarlyStopping":{EarlyStopping:[7,1,1,""],EarlyStoppingTTest:[7,1,1,""]},"UCTB.train.EarlyStopping.EarlyStopping":{__best:[7,2,1,""],__p:[7,2,1,""],__patience:[7,2,1,""],__record_list:[7,2,1,""],stop:[7,3,1,""]},"UCTB.train.EarlyStopping.EarlyStoppingTTest":{__best:[7,2,1,""],__p_value_threshold:[7,2,1,""],__record_list:[7,2,1,""],__test_length:[7,2,1,""],stop:[7,3,1,""]},"UCTB.train.MiniBatchTrain":{MiniBatchFeedDict:[7,1,1,""],MiniBatchTrain:[7,1,1,""],MiniBatchTrainMultiData:[7,1,1,""]},"UCTB.train.MiniBatchTrain.MiniBatchFeedDict":{get_batch:[7,3,1,""],restart:[7,3,1,""],shuffle:[7,3,1,""]},"UCTB.train.MiniBatchTrain.MiniBatchTrain":{get_batch:[7,3,1,""],restart:[7,3,1,""],shuffle:[7,3,1,""]},"UCTB.train.MiniBatchTrain.MiniBatchTrainMultiData":{get_batch:[7,3,1,""],restart:[7,3,1,""],shuffle:[7,3,1,""]},"UCTB.utils":{multi_threads:[8,0,0,"-"],st_map:[8,0,0,"-"]},"UCTB.utils.multi_threads":{multiple_process:[8,4,1,""],task:[8,4,1,""]},"UCTB.utils.st_map":{st_map:[8,4,1,""]},UCTB:{dataset:[2,0,0,"-"],evaluation:[3,0,0,"-"],model:[4,0,0,"-"],model_unit:[5,0,0,"-"],preprocess:[6,0,0,"-"],train:[7,0,0,"-"],utils:[8,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function"},terms:{"1080ti":[17,18],"10h":24,"7500m":24,"8700k":[17,18],"\u4e00\u5171\u6709428\u79cdpoi":26,"\u4e09\u4e2a\u57ce\u5e02\u7684poi\u6570\u91cf\u4e0echeck":26,"\u4e0a\u7684period":[],"\u4e4b\u95f4":26,"\u5206\u522b\u4e3a\u5de5\u4f5c\u65e5":26,"\u5230":26,"\u534a\u5f84\u4e3a50km\u7684poi\u6570\u91cf":26,"\u5373\u6bcf\u4e2a\u7ad9\u70b9\u6709428\u7ef4\u7279\u5f81":26,"\u5373\u6bcf\u4e2a\u7ad9\u70b9\u7684\u7279\u5f81\u7ef4\u5ea6\u4e3a48":26,"\u53f3\u4fa7\u4e3a":26,"\u5404\u4e2a\u81ea\u884c\u8f66\u7ad9\u70b9\u9644\u8fd11km\u7684checkin\u603b\u6570\u91cf":26,"\u5747\u503c":24,"\u57ce\u5e02":26,"\u591a\u6b21\u5b9e\u9a8c\u7ed3\u679c":24,"\u5b9e\u9a8c\u7f16\u53f7":24,"\u5de5\u4f5c\u65e511707":26,"\u5de5\u4f5c\u65e52549":26,"\u5de5\u4f5c\u65e56049":26,"\u5de6\u4fa7\u4e3a":26,"\u5e73\u5747\u8017\u65f6":24,"\u5f85\u8865\u5145":[],"\u6240\u4ee5\u4ec5\u4f7f\u7528closeness\u6548\u679c\u5f88\u5dee":[],"\u6309\u7167\u5de5\u4f5c\u65e5\u548c\u8282\u5047\u65e5\u5206\u5f00":26,"\u65e5\u5747check":26,"\u65f6\u95f4\u8303\u56f4":26,"\u6682\u65f6\u53ea\u8dd1\u4e864\u6b21":24,"\u6700\u7ec8\u7ed3\u679c":24,"\u6807\u51c6\u5dee":24,"\u6a21\u578b\u7248\u672c\u542b\u4e49":24,"\u6bcf\u4e2a\u7ad9\u70b9\u7edf\u8ba1\u9644\u8fd11km\u51fa\u73b0\u7684poi\u7c7b\u578b":26,"\u6bcf\u6b21\u5b9e\u9a8c\u8017\u65f6":24,"\u6bcf\u6b21\u5b9e\u9a8c\u8017\u65f6\u7ea6":24,"\u7c97\u7565\u8ba1\u7b97\u6240\u6709\u7ad9\u70b9\u9644\u8fd11km\u7684checkin\u6570\u91cf\u603b\u548c":26,"\u7ea2\u8272\u4ee3\u8868\u5dee\u4e8efinetun":26,"\u7eff\u8272\u7684\u70b9\u8868\u793atransfer\u6548\u679c\u597d\u4e8efinetun":26,"\u8282\u5047\u65e511358":26,"\u8282\u5047\u65e52692":26,"\u8282\u5047\u65e55450":26,"\u8282\u5047\u65e5\u768424\u5c0f\u65f6checkin":26,"\u8ba1\u7b97":26,"\u8ba1\u7b97\u57ce\u5e02\u4e2d\u5fc3\u4e3a\u539f\u70b9":26,"\u8bad\u7ec3\u6837\u672c\u6570\u91cf":26,"case":27,"class":[2,4,5,6,7,13,27],"default":[2,4,5,6,7,10,18],"final":[5,27],"float":[2,4,5,7],"function":[2,4,5,27],"import":[11,14,23,27],"in\u6570\u91cf":26,"int":[2,4,5,6,7,27],"long":[10,27],"new":[7,10,17,27],"null":7,"poi\u6570\u91cf":26,"public":27,"return":[2,4,5,6,7,27],"rmse\u503c":24,"short":[10,17,18,19,20,27],"static":[4,5,6,7],"super":27,"trend\u7279\u5f81\u6027\u66f4\u5f3a":[],"true":[2,4,5,7,11,14,24,27],"try":[23,27],"veli\u010dkovi\u0107":10,And:[11,12,27],For:[7,27],His:[],Its:[2,7],Not:[],The:[2,4,5,6,7,10,11,12,17,18,27],There:7,These:[4,27],Use:[9,17,18],Used:2,Using:[16,23],With:27,__best:7,__init__:[5,27],__p:7,__p_value_threshold:7,__patienc:7,__record_list:7,__test_length:7,_graph:27,_input:27,_op:27,_output:27,about:[9,17,18,19,20,27],accept:27,accord:[2,4,6,7,27],account:27,accus:27,acquir:27,activ:[4,5],adam:4,adamoptim:27,add:[5,18],add_ga_layer_matrix:5,add_gc_lay:5,add_multi_gc_lay:5,add_residual_ga_lay:5,add_summari:5,added:5,adding:5,addit:5,adf_test:4,adjacent_matrix:5,adjacent_to_laplacian:5,adjust:4,after:5,aggreg:[5,10],alajali:16,algorithm:[4,16],all:[2,4,10,21,24,27],almost:5,alpha:5,alreadi:[11,12],also:[2,5,11,12,27],alwai:16,among:16,amp:10,analysi:4,api:[4,9,17,18,19,20,27],apidoc:30,append:[5,7,23,27],applic:[10,27],approxim:[4,5],apr:26,arbitrari:2,architectur:[16,27],area:16,arg:[4,5,10],argu:4,argument:5,arima:[0,1,10,11,13,17,18,19,20,29],arma:4,arrai:[5,23,27],arrang:2,array_lik:4,art:[11,13],as_default:27,assist:28,assum:[5,16],attent:[4,5,10],attention_merge_weight:5,attribut:[2,4,10,17,18,19,20],aug:28,augment:4,auto:[4,5],auto_load_model:5,automat:5,autoregress:4,averag:[4,7,17,18,19,20],axi:[4,23,27],back:[12,13,16,17,18,19,20,27],base:[2,4,5,6,7,10,11,13,27],baselin:21,basemodel:[0,1,4,27,29],bash:[11,12],basi:16,basic:[2,27],basiclstmcel:27,batch64:[],batch:[4,5,7,27],batch_siz:[5,7,10,27],batchsiz:24,befor:5,begin:27,beij:[10,18],being:16,bengio:10,best:[7,10],better:[7,27],between:[2,5,10],bia:5,bidirect:16,big:28,bike:[4,9,14,18,23,27],bike_chicago:10,bike_dc:10,bike_nyc:10,bikeshar:27,bin:[11,12],binar:5,bool:[2,4,5,7],boost:[4,16],borrow:27,both:[4,5,11,13,16],build:[2,4,5,9,13,14,30],build_graph:2,build_ord:[2,8],built:[4,5,17,18],cach:5,cache_volum:5,calcul:[4,5,6],call:[5,6,7,27],callabl:4,can:[2,4,5,6,7,10,11,12,13,27],captur:[4,16],casanova:10,castleliang:4,cell:5,certain:[2,5],chai:[10,28],chang:[6,7],channel:5,chapter:[11,13],charg:[9,11,27],chargest:10,chargestation_beij:10,chebyshev:[4,5],check:[7,27],checkin_sim:2,checkin_sim_sd:2,chen:[16,28],chengdu:[10,19],chenliyu:[],chenliyue2019:28,chicago:[10,17,18,27],china:[],chinesecalendar:[11,12],chongq:[10,20],circl:5,citi:[2,10,11,14,17,18,23,24,27],citywid:[4,16],ck1manozn0edb1dpmvtzle2cp:2,cl_decay_step:4,classic:27,clip:4,close:[2,4,5,6,10,16,18,27],closeness_featur:[4,11,14,23],closeness_len:[2,4,6,11,14,23,27],closenss:6,cmd:30,code:[4,5,10,17,18,30],code_vers:[4,5,27],codevers:24,coeffici:[2,5],collect:[17,18,27],color:[17,18],com:28,combin:16,common:[],commonli:27,compar:[11,27],compat:27,complet:27,complex:25,compon:[4,16,27],compos:[4,6,16],comput:[5,12,13,27,28],compute_output_shap:5,concat:[4,10,25],concaten:[2,23,27],concret:27,confid:28,config:[11,14],consecut:[2,4,6],consid:27,considerd:4,consist:[2,4,7,16],constant:16,construct:[2,27],constructor:27,contact:[],contain:[2,11,13,27],content:[0,29],continu:27,contrib:27,contribut:27,contribute_data:2,contributor:9,conv_filt:[4,10],convent:[11,13],converag:[17,18],converg:[23,27],convert:6,convolut:[4,5,10],coordin:2,correl:[2,4,5,10,11,14,16,17,18,19,20,24],correlation_adjac:5,correspond:[2,4,5],cosin:26,could:27,count:27,cover:27,cpt:[],cpu:[17,18],creat:2,crowd:[4,16],crowdsens:[],cse:28,cucurul:10,cummin:27,current:[4,5,7,10,11,13,27],custom:4,dai:[2,4,6,16],daili:[4,17,18,19,20],daily_slot:[2,6],data:[2,4,5,6,7,8,10,13,14,17,18,19,20,27,28],data_dir:2,data_load:[0,1,11,14,23,27,29],data_rang:[2,27],datafram:4,datarang:24,dataset:[0,1,4,5,9,14,17,18,19,20,23,24,29],date:6,date_str:6,dateutil:[11,12],dcgru:10,dcgrucel:5,dchai:28,dci:10,dcrnn:[0,1,10,11,13,29],dcrnn_cell:[0,1,29],debug:4,decid:7,decim:5,decis:16,decod:[4,16],deep:[4,5,10],deeper:[17,18],deepst:[0,1,11,13,29],def:27,defin:27,degre:[4,5],demand:[4,16,27],denorm:6,dens:[4,10,18,19,20,25,27],dense_unit:[],denseunit:[18,19,20,24],depart:28,depend:[4,16],depth:[4,10],describ:[4,16],design:27,desir:27,detail:[10,11,13,27],determin:5,devic:24,dichai:[11,12],dickei:4,dict:[2,4,5,6,7,27],dictionari:[6,7],didi:9,didi_chengdu:10,didi_xian:10,differ:[4,9,11,16],differenc:4,diffus:[4,10],dim:27,dimens:[2,4,5,6,7,27],direct:26,directli:[2,7],directori:[2,4,5],disabl:5,disk:5,distacn:5,distanc:[2,5,10,17,18,19,20,24],distance_adjac:5,distribut:[4,27],distribute_list:8,divid:[6,7],divis:27,dnn:[4,16],docker:9,docstr:30,domain:[4,16],driven:[4,10,16],drop:4,dropout:4,dropout_r:4,dtype:[5,27],dump:27,dure:[2,5],dynam:[4,16],dynamic_rnn:27,each:[4,5,7,27],earli:[4,5,7,16,27],earlier:2,early_stop_length:5,early_stop_method:5,early_stop_pati:5,earlystop:[0,1,5,29],earlystoppingttest:7,earth:5,easi:4,edu:28,effort:27,either:27,electr:27,element:[5,6],els:[4,5],email:28,embed:2,empir:16,empti:2,encod:[4,16],end:[2,4,16,27],engin:28,enough:7,ensur:21,epoch:[5,24,27],epsilon:4,equal:[4,5,6,7],error:23,eslength:24,especi:21,estimat:10,etc:27,euclidean:16,eval_metr:4,evalu:[0,1,4,9,10,14,17,18,23,27,29],evaluate_loss_nam:5,everi:[2,4,6],exact:[2,4,6],exampl:27,exce:7,except:[5,23],exist:5,expand:[],expand_dim:27,expect:7,experi:[9,21,27],explain:5,explan:[17,18,19,20,24],explicitli:16,extend:27,extern:[2,4,16],externai_dim:4,external_dim:[2,4],external_featur:4,externalfeatur:27,extra:16,extract:[2,27],factor:[4,16,27],fail:23,fals:[2,4,5,7,23,24,27],far:[4,16],featur:[2,4,5,6,7,10,18,25],feature_length:6,feature_step:6,feature_strid:6,feb:28,februari:16,feed_dict:[6,7],few:[],file:[2,4,5,10],file_nam:8,filenam:[4,5],fill:27,filter:4,final_st:27,find:27,finetun:26,finish:23,first:[7,11,12,27],firstli:27,fit:[4,5,11,14,23,27],fix:27,flag:5,flexibl:[11,13],float32:[5,27],flog:5,flow:[4,10,16,18,27],follow:[10,11,12,13,17,18,19,20,27],forecast:[4,10,16],forecast_step:[4,23],forget:27,format:[2,5,27],former:[2,4,6],found:[5,10,27],four:10,frame:4,framework:16,franc:[],fratur:4,free:5,from:[2,4,5,7,9,12,14,16,23,28],fuller:4,further:2,fuse:[4,16],fusion:[4,16],futur:[4,16,27],gal:[4,5,10,25],galhead:[18,19,20,24],galunit:[18,19,20,24],gate:[5,10],gattel:[18,19,20],gaussianhmm:4,gbrt:[10,17,18,19,20],gc_output:5,gc_rate:4,gcl:5,gcl_k:4,gcl_l:4,gclstm:[4,10,17,18,19,20,25],gclstm_layer:4,gclstmcell:5,gcn:[4,5],gcn_k:[4,5],gcn_l:5,gcn_layer:4,gener:[4,6,7,16],general_move_sampl:6,generaliz:10,geng:[4,16],geo:4,geograph:5,geoman:[0,1,29],ger:27,get:7,get_a_cel:27,get_batch:7,get_ord:4,github:4,give:4,given:6,gll:[18,19,20,24],global:[4,16],global_featur:4,global_step:5,gmail:28,good:27,got:[],gpu:[4,5,11,12,17,18],gpu_devic:[4,5,27],gputil:[11,12],gradient:[4,16],graph:[2,4,5,10,11,14,17,18,19,20,25,27],graph_merg:4,graph_merge_gal_num_head:4,graph_merge_gal_unit:4,graph_nam:2,graphbuild:5,graphmodellay:[0,1,29],great:5,greater:5,grid:[4,10,27],gridlatlng:27,gridtrafficload:2,ground:4,group:[9,24],gru:4,gtx:[17,18],hail:[4,16],handl:[5,27],has:2,have:[2,4,6,7,11,12,17,18],haversin:5,head:[4,5,18,19,20],height:4,help:[6,27],here:[10,27],hidden:[4,5,18,19,20,27],high:28,highest:[4,5],highest_protocol:27,highli:[11,12],him:[],his:[],hisrori:4,histor:[4,27],histori:[2,4,6],hm_obj:23,hmm:[0,1,11,13,17,18,19,20,27,29],hmm_kernal:4,hmmlearn:[4,11,12,16],hoel:16,hold:5,homepag:[12,13,16,17,18,19,20,27,28],hong:28,hour:[17,18],how:[2,27],html:30,http:[11,12],hub:[11,12],hypothesi:7,ident:7,ignor:4,illustr:27,imag:15,implement:[4,5,10,27],includ:[4,5,6,16,27],incorpor:[4,16],independ:7,index:2,indic:2,inform:[5,16,27],inherit:[5,27],init:[4,5,11,14],init_var:[4,5,27],initi:[4,5,27],input:[2,4,5,6,7,16,27],input_dim:[4,5,27],input_shap:5,input_step:[4,27],input_transform:4,inspect:27,instal:9,instanc:[4,16],instead:5,institut:[],instruct:[],integ:[2,4,5],interact:[2,5,10,17,18,19,20,24],interaction_adjac:5,interaction_matrix:5,interatct:2,interest:27,interl:[17,18],intern:7,intersect:16,interv:7,introduc:10,introduct:9,is_train:2,is_valid_d:6,is_work_day_america:[2,6],is_work_day_china:6,iter:4,its:[2,6,10],jin:28,jinxu:28,jpg:[],jul:28,juli:16,junbo:4,keep:[4,5],kei:[6,7,9,27],kera:[5,11,12],kernal_s:10,kernel:4,kernel_initi:5,kernel_s:[4,10],keyword:5,kind:4,know:[17,18,19,20],knowledg:27,kong:28,kth_cheby_ploi:5,kwarg:[2,3,4,5],lab:28,label:4,lag:4,land:[4,16],laplac:5,laplace_matrix:[11,14],laplacian:[2,5],laplacian_matrix:5,larger:[2,5],last:[7,23],lat1:5,lat2:5,lat:[8,27],lat_lng_list:5,late:[4,16],later:2,latest:[2,18],latitud:[2,5],layer:[4,5,10,18,19,20,27],leaky_relu:5,learn:[4,10,11,12,19,20,27],learner:4,least:27,len:4,length:[2,4,5,6,7,18,23,27],less:4,let:27,level:4,ley:28,leyewang:28,liang:[4,16],librari:4,life:[17,18,19,20],light:8,like:5,likelihood:4,limit:7,line:[2,10,20,24],linear:27,link:5,lio:10,list:[2,4,5,6,7],liu:[10,16],live:5,liyaguang:4,liyu:28,lng:[8,27],load:[5,27],load_event_scalar:5,loader:[2,11,14,27],local:[4,16,27],local_featur:4,locat:5,locker:8,log:5,logic:5,lon1:5,lon2:5,longitud:[2,5],look:27,loss:[5,27],lot:27,lstm:[4,5,10,16,17,18,19,20,27],lstm_hidden_unit:[],lstm_layer:4,lstm_unit:4,lstmc:10,lstmcell:5,lstmunit:[18,19,20,24],lucktroi:4,machin:[4,16],mai:[2,4],main:[2,10],major:[4,16,28],make:[4,16,27],make_concat:2,manag:9,mani:[2,5,27],manual_summari:5,map:[10,17,18],mapbox:2,mape:[3,24],mape_grid:3,mark:10,markov:27,master:[],match:5,matplotlib:27,matric:[2,4,6],matrix:[4,5],max:[2,4,5],max_ar:4,max_d:4,max_depth:[4,23,27],max_diffusion_step:[4,5],max_epoch:[5,27],max_lag:4,max_ma:4,max_to_keep:[4,5,27],maximum:4,mean:[4,17,18,19,20],mechan:[4,16,27],memori:[5,10,17,18,27],merg:[4,25],merge_gal_unit:5,meta:10,meta_info:8,meteorolog:[4,16],meter:[2,5],method:[2,4,5,6,7,21,25,27],metric:[0,1,4,11,14,23,27,29],metro:[9,24,27],metro_chongq:10,metro_shanghai:10,mgcn:[4,5,10],million:27,min:2,min_max_denorm:[6,11,14,27],min_max_norm:6,mini:5,minibatchfeeddict:7,minibatchtrain:[0,1,29],minibatchtrainmultidata:7,minim:27,minut:[2,27],mobil:[],mode:4,model:[0,1,2,5,7,9,10,13,17,18,19,20,23,29],model_dir:[4,5,27],model_obj:23,model_param:2,model_r:4,model_unit:[0,1,4,9,27,29],modifi:10,modul:[0,16,29],month:[2,5,27],month_num:2,monthli:2,more:[4,5,10,17,18,19,20,27],move:4,move_sampl:6,movesampl:6,multi:[4,5,10,17,18,19,20,25],multi_thread:[0,1,29],multipl:16,multiple_process:8,multirnncel:27,must:[4,5,27],my_dataset:27,my_lstm:27,my_model:27,mymodel:27,n_decoder_hidden_unit:4,n_encoder_hidden_unit:4,n_estim:[4,23,27],n_iter:[4,23,27],n_job:8,n_stacked_lay:4,naiv:[4,5],name:[2,5,10,25,27],ndarrai:[2,4,5,6,7],necessari:27,need:[2,4,5,6,27],neighbor:[2,20],network:[4,5,10,17,18,19,20],neural:[2,4,5,10],new_valu:7,newest:7,next:27,nni:[11,12],node:[2,4,5,23,27],node_monthly_interact:2,node_num:[2,4],node_station_info:2,node_traff:2,nodetrafficload:[2,11,14,23,27],non:16,none:[2,4,5,27],norm:4,normal:[2,4,6,11,14,23,24,27],notat:24,note:[2,4],noth:5,nov:28,novemb:[10,16],now:[2,4,6,27,28],num:27,num_compon:[4,23,27],num_conv_filt:4,num_dense_unit:4,num_diffusion_matrix:4,num_featu:5,num_featur:5,num_graph:[4,5,11,14],num_head:5,num_hidden_unit:4,num_lay:27,num_nod:[4,5,11,14],num_proj:5,num_residual_unit:[4,10],num_rnn_lay:4,num_rnn_unit:4,num_stat:27,num_unit:[5,27],number:[2,4,5,6,7,10,17,18,19,20,27],numpi:[5,6,11,12,23,27],nvidia:[11,12,17,18],nyc:[4,10,11,14,17,18,23,27],obj:10,object:[2,4,5,6,7,11,14,23,27],observ:4,obviou:10,octob:16,often:16,onc:7,one:[2,4,5,6,7,27],onli:[4,7,10,18,27],op_nam:5,open:[11,13,27],oper:[4,27],operation1_nam:5,operation2_nam:5,ops:5,optim:[4,5,27],optimizer_nam:4,option:[2,4],order:[4,5,7,23],org:[11,12],orgin:6,other:[2,5,9,11,16,27],otherwis:[2,4,5,7],our:[5,10,16],output:[4,5,27],output_activ:4,output_dim:[4,27],output_nam:[5,11,14],output_s:5,output_step:[4,27],output_tensor1_nam:5,output_tensor1_valu:5,output_tensor_name1:5,output_tensor_name2:5,outputs_dict:5,overal:26,overwrit:5,own:[2,5,9,13],p_test:23,p_value_threshold:7,packag:[0,9,11,12,13,16,27,29],page:[11,12],pair:[2,6,7],panda:[11,12,16],paper:16,param:[5,17],paramet:[2,4,5,6,7,8,10,17,18,19,20,25,27],pari:[],part:[4,16],partition_func:8,pass:4,past:27,path:[2,27],patienc:[7,24],pattern:16,pearson:[2,5],peke:28,penultim:[18,19,20],per:5,perform:[5,10,27],period:[2,4,6,10,16,18,27],period_featur:[4,11,14,23],period_len:[2,4,6,10,11,14,23,27],pickl:[2,27],piec:[2,4,6],pip:[11,12],pkl:27,pkl_file_nam:27,pku:28,placehold:27,pleas:[17,18,19,20],plot:27,plt:27,poi:27,poi_sim:2,point:[4,5],polynomi:[4,5],poor:10,popular:27,portal:27,posit:2,postdoc:[],predict:[2,3,4,5,6,10,13,14,18,23,27],prediction_test:23,preprocess:[0,1,9,17,18,19,20,27,29],preprocessor:[0,1,29],presenc:4,present:4,preserv:[],preset:27,previou:4,print:[5,11,14,23,27],privaci:[],problem:27,process:[2,4,16,25],produc:5,professor:28,project:[9,27],properti:[4,5,16],proport:2,protect:[],protocol:27,provid:[5,11,13,27],pull:[11,12],pyplot:27,python:[5,10,11,12],pytorch:4,pyyaml:[11,12],quick:9,quickstart:[4,27],random:16,rang:[2,23,27],rate:[4,19,20],ratio:[2,4,6,26],ratio_list:6,ration:5,raw:[5,6],real:[17,18,19,20],realiz:[10,27],receiv:[],recent:27,recognit:16,recommend:[11,12],record:[2,6,7,10,17,18,19,20,27],recurr:[4,5,10],reduce_func:8,reduce_mean:27,refer:[4,9,10,12,16,17,18,19,20,27],reg:[4,23,27],region:16,regular:5,rel:2,relat:27,releas:[5,27],remain:[17,18],remov:[17,18,19,20],repres:5,represent:5,requir:[11,12,16,27],research:[],reserv:4,reshap:[23,27],residu:[4,5,16],residual_unit:10,resnet:[4,10,11,13],resourc:15,respect:[17,18,27],restart:7,result:[2,4,5,6,9,11,14,16,18,19,20,26,27],return_output:5,reus:5,revers:27,rgal:25,ride:[4,10,16,17,18,19,20],rideshar:27,rmetfc:2,rmse:[3,4,11,14,23,24,27],rmse_grid:3,rnn:[4,5,27],rnn_cell:27,rnn_cell_impl:5,rnncell:5,romero:10,root:4,run:[10,11,12,27,30],runtim:[11,12],sai:27,same:[2,4,5,6],sampl:[4,7,16],sarima:[],save:[4,5,27],save_model:5,save_model_nam:5,saver:[4,5],scalar:5,scalar_nam:5,scenario:27,schedul:16,schmidhub:27,scienc:28,scikit:[11,12],scipi:[4,11,12],scratch:7,sd_param:2,search:17,season:[4,16],seasonal_ord:[4,23],second:[2,27],see:[2,4,7,27],select:2,self:[2,27],sensor:4,sensori:4,sept:26,seq_len:4,sequenc:[4,5,27],sequence_length:[5,6,7,11,14,27],seri:[4,27],serial:4,session:[5,7],set:[2,4,5,7,9,17,18,19,20,27],seven:[2,4,6],sever:7,sgd:4,shahabi:[10,16],shall:2,shanghai:[10,20,24],shanghaiv1:24,shape:[2,4,5,11,14,23,27],share:10,share_queu:8,shawnwang:4,should:[2,4,5,7,27],show:[10,17,18,19,20,27],shuffl:[5,7],shuffle_data:5,side:7,sigmoid:4,signific:[17,18,19,20],silent:4,similar:[2,4,16,26],simpl:[16,27],simpli:4,sinc:[4,17,18,19,20,27],singl:[2,10],singlegraph:[17,18],size:[4,5,7],skip:[11,12],slot:[2,4,6,27],small:7,smaller:[5,7,17,18,19,20],softmax:5,softwar:28,sort:2,sourc:[17,18],space:[],span:[7,10,17,18,19,20],spars:4,spatial:[4,5,10,11,13,27],spatio:[4,10,16],spatiotempor:4,specif:[],specifi:[2,4,5,10,27],sphinx:30,split:[2,4,6],split_data:6,split_feed_dict:6,split_timestep:4,splitdata:[6,27],sqrt:27,squar:27,squarederror:[4,23],squeez:27,src:[],st0:24,st_map:[0,1,2,29],st_method:4,st_mgcn:[0,1,10,17,18,29],st_movesampl:6,st_resnet:[0,1,5,10,29],st_rnn:[0,1,29],st_sim1_0:24,st_sim_0:24,stack:4,stacked_cel:27,stacked_output:27,stage:[17,18,19,20],start:[2,9,27],state:[5,11,13,16],state_is_tupl:27,state_s:5,station:[4,9,11,17,19,20,23,27],station_index:23,station_numb:[2,11,14,23,27],stationinfo:[2,27],statist:16,statsmodel:[11,12,16],statu:27,step:[2,4,5,6],stmeta:[0,1,5,9,13,17,18,19,20,24,29],stmeta_master_bik:[17,18],stmeta_obj:[10,11,14],stmeta_v0:10,stmeta_v1:10,stmeta_v2:10,stmeta_v3:10,stop:[5,7,27],store:[2,4,5,27],str:[2,4,5],stream:2,string:[2,4,5],strnn:4,strongli:27,structur:[4,11,13,16],student:28,style:[2,8],subclass:27,submodul:[0,1,29],subpackag:29,subscript:5,subset:2,sudpari:[],supplementari:2,support:[5,10,11,13,27],system:[10,16],tail:7,take:[7,27],tanh:[4,5],target:[3,4,6,7,11,14,27],target_len:4,target_length:[2,6,27],target_nod:27,target_nodz:27,target_st:27,task:[4,8],task_func:8,td_data_length:2,td_param:2,tech:4,techniqu:[5,10],technolog:28,telecom:[],temp_merge_head:[],temp_merge_unit:[],temperatur:4,tempor:[2,4,5,6,10,11,13,25],temporal_merg:4,temporal_merge_gal_num_head:4,temporal_merge_gal_unit:4,temporam:4,tensor:[5,27],tensorboard:5,tensorflow:[4,5,16,27],tensorshap:5,term:[10,27],test:[2,4,5,7,10,11,14,23,27],test_clos:[2,11,14,23,27],test_i:[2,11,14,23,27],test_period:[2,11,14,23,27],test_predict:23,test_prediction_collector:23,test_ratio:[2,27],test_rms:[23,27],test_sequence_len:[11,14,27],test_trend:[2,11,14,23,27],test_x:27,than:[2,4,5,7,17,18,19,20],thei:7,them:[5,16,27],theoret:16,therefor:27,thi:[4,5,6,7,11,12,27,30],those:7,three:[4,10,16],threshold:[5,7,11,14,18,19,20,23,24,27],threshold_correl:2,threshold_dist:2,threshold_interact:2,thu:5,tile:[],till:28,time:[2,4,5,6,7,10,17,18,19,20,27],time_fit:[2,6,27],time_index:23,time_rang:[2,27],time_sequ:[4,23],time_seri:4,time_slot_num:[2,4],time_util:[0,1,29],timefit:[2,27],timerang:[2,27],timestep:[4,27],tk1:5,tk2:5,togeth:5,toi:27,too:2,toolbox:13,total:4,total_sens:4,traffic:[4,10,16,18,27],traffic_data:[5,27],traffic_sim:2,traffic_sim_fak:2,trafficgrid:27,trafficmonthlyinteract:[2,27],trafficnod:[2,27],train:[0,1,2,4,5,9,13,14,17,18,23,24,27,29],train_clos:[2,11,14,23,27],train_data_length:2,train_i:[2,11,14,23,27],train_loss:27,train_op:[5,27],train_period:[2,11,14,23,27],train_sequence_len:[11,14,27],train_time_slot_num:2,train_trend:[2,11,14,23,27],train_x:27,trainabl:[5,17,18],trainable_var:27,trainbl:27,traindai:24,transfer:[2,26],transferdataload:2,transform:[5,27],transpos:23,tree:[4,16],trend:[2,4,6,10,16,18,27],trend_featur:[4,11,14,23],trend_len:[2,4,6,10,11,14,23,27],trigger:7,truth:4,tupl:5,turn:5,tutori:[5,9,13],two:[2,4,5,7,16],type:[2,4,5,6,7],ubiquit:[],uctb:[0,13,14,23,30],undergradu:28,uniqu:[4,16],unit:[4,5,10,13,16,18,19,20],univari:4,univers:28,unviers:[],updat:30,upgrad:[11,12],urban:[13,27],usag:27,use:[2,4,5,10,11,12,13,16,17,18,27],use_bia:5,use_curriculum_learn:4,use_gc_for_ru:5,used:[2,4,5,6,10,16,17,18,19,20,27],useful:27,uses:[19,20,27],using:[4,5,9,16,23],ust:28,util:[0,1,9,29],val:[],val_loss:[5,27],valid:[4,5],validate_ratio:[5,27],valu:[4,5,6,7,18,19,20,24,27],variabl:[7,17,18,27],variant:10,variou:[5,27],vector:6,vehicular:16,verbos:[4,5],version:[4,5,9,11,12,27],view:[17,18],visual:[10,17,18,27],walk:16,wang:[10,16,28],washington:27,weather:27,week:[2,4,6],weekli:4,weight:5,welcom:27,well:[11,13,27],wen:16,wenji:28,wget:[11,12],when:[2,4,5,7,27,30],where:[4,5],whether:[5,7],which:[4,5,16,27],whole:[5,27],whose:[7,27],wide:16,width:4,william:16,wind:4,window:[17,18],with_lm:[2,23,27],with_tp:[2,27],without:6,work:4,workday_pars:2,wors:7,wrapper:27,www:[11,12],xgboost:[0,1,10,11,12,13,17,18,19,20,27,29],xian:10,xidian:28,yaguang:4,yang:[10,16,28],year:[],yet:[],yml:10,york:[10,17,27],yoshal:4,you:[5,11,12,13,27],your:[2,5,9,13],yuxuan:4,yyyi:[2,27],zero:[23,27],zhang:[4,16],zheng:16,zhou:16,zoom:[2,8]},titles:["5. API Reference","UCTB package","5.1. UCTB.dataset package","5.5. UCTB.evaluation package","5.4. UCTB.model package","5.3. UCTB.model_unit package","5.2. UCTB.preprocess package","5.6. UCTB.train package","5.7. UCTB.utils package","Welcome to UCTB\u2019s documentation!","6. Results on different datasets","Urban Computing ToolBox","2. Installation","1. Introduction","3. Quick start","&lt;no title&gt;","Currently Supported Models","Experiments on bike traffic-flow prediction","Experiments on Charge-Station demand station","Experiments on DiDi traffic-flow prediction","Experiments on subway traffic-flow prediction","Purpose","&lt;no title&gt;","Quick Start with HM (Historical Mean)","Stable Test Records","Different versions of STMeta","Check-In\u4e0ePOI\u6570\u636e\u5904\u7406\u65b9\u6cd5","4. Tutorial","7. About us (UCTB Group)","UCTB","&lt;no title&gt;"],titleterms:{"\u4f7f\u7528check":26,"\u4f7f\u7528poi\u4fe1\u606f\u8fdb\u884c\u76f8\u4f3c\u7ad9\u70b9\u5339\u914d":26,"\u5206\u6790transfer\u6548\u679c\u5728\u57ce\u5e02\u4e2d\u7684\u5206\u5e03":26,"\u6570\u636e\u8be6\u60c5":26,"\u7279\u5f81\u76f8\u4f3c\u5ea6\u65b9\u6cd5":26,"\u7279\u5f81\u8ba1\u7b97\u65b9\u6cd5":26,"class":11,"in\u4e0epoi\u6570\u636e\u5904\u7406\u65b9\u6cd5":26,"in\u6570\u636e\u8fdb\u884c\u76f8\u4f3c\u7ad9\u70b9\u7684\u5339\u914d":26,"poi\u7279\u5f81\u8ba1\u7b97\u65b9\u6cd5":26,Adding:[19,20],Not:[],Use:[11,27],about:28,api:[0,11],arima:[4,16,23],attent:16,autoregress:16,averag:16,base:16,basemodel:5,beij:24,bike:[10,11,17],build:[11,24,27],charg:[10,18],chargest:24,check:26,chengdu:24,chicago:26,citi:26,close:[17,19,20],comput:11,content:[1,2,3,4,5,6,7,8],contribut:11,contributor:28,convolut:16,current:16,data:[11,16],data_load:2,dataset:[2,10,11,27],dcrnn:[4,16],dcrnn_cell:5,deep:16,deepst:[4,16],delai:11,delet:17,demand:[11,18],didi:[10,11,19,24],differ:[10,25],diffus:16,docker:[11,12],document:9,earlystop:7,evalu:[3,11],exampl:11,experi:[10,11,17,18,19,20],featur:[17,19,20,27],finish:[],flow:[11,17,19,20],from:[11,27],full:11,futur:17,geo:16,geoman:[4,16],graph:[16,24],graphmodellay:5,group:28,hidden:16,histor:[16,23],hmm:[4,16,23],instal:[11,12],integr:16,introduct:[11,13],kei:28,latest:17,learn:16,level:16,ley:[],manag:28,markov:16,mean:[16,23],metric:3,metro:[10,11],mgcn:16,minibatchtrain:7,mode:[19,20],model:[4,11,14,16,24,27],model_unit:5,modelunit:11,modul:[1,2,3,4,5,6,7,8],move:16,multi:16,multi_thread:8,multipl:27,network:16,neural:16,nyc:26,onli:17,order:11,other:14,own:[11,27],packag:[1,2,3,4,5,6,7,8],paramet:24,period:[19,20],predict:[11,16,17,19,20],preprocess:6,preprocessor:6,project:28,purpos:21,quick:[11,14,23],record:24,recurr:16,refer:[0,11],regress:27,resnet:16,result:[10,17],search:10,sensori:16,seri:16,set:10,singl:27,space:10,spatial:16,spatiotempor:16,st_map:8,st_mgcn:4,st_resnet:4,st_rnn:5,stabl:24,start:[11,14,23],station:[10,18],statist:10,step:[11,12],stmeta:[4,10,11,14,16,25],submodul:[2,3,4,5,6,7,8],subpackag:1,subwai:20,support:16,target:26,tempor:[16,27],tensorflow:[11,12],test:24,time:16,time_util:6,toolbox:11,traffic:[11,17,19,20],train:[7,11],tutori:[11,27],uctb:[1,2,3,4,5,6,7,8,9,11,12,27,28,29],unit:11,urban:11,using:[11,27],util:8,version:[10,17,25],wang:[],welcom:9,xgboost:[4,16,23],xian:24,yet:[],your:[11,27]}})