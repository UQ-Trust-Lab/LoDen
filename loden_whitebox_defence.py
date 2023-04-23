from models import *
from constants import *
import pandas as pd
import numpy as np
import copy, os, random
torch.set_printoptions(precision=4, sci_mode=False)



class Organizer():
    def __init__(self, train_epoch=TRAIN_EPOCH):
        self.set_random_seed()
        self.reader = DataReader()
        self.target = TargetModel(self.reader)
        self.bar_recorder = 0
        self.last_acc = 0
        self.attack_round = 0

    def set_random_seed(self, seed=GLOBAL_SEED):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def honest_node_vector_defense(self, logger, adaptive=False, record_process=True, record_model=False, plot=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
                                                "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                                                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        attacker_success_round = []

        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(), DEFAULT_AGR)
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info("Dataset is {}".format(DEFAULT_SET))
        logger.info("Member ratio is {}".format(BLACK_BOX_MEMBER_RATE))

        # Initialize data for one honest node, every round evaluate 2 samples.
        member_total_samples = self.reader.get_honest_node_member()
        all_member_samples = self.reader.get_train_set(0) # Get all member samples
        last_batch = self.reader.get_last_train_batch() # Get last batch
        number_of_round = len(member_total_samples) // 5 # Get number of rounds
        member_total_samples = member_total_samples[0:100] # Get first 100 samples for testing

        # Initialize attacker
        logger.info(len(member_total_samples)//2)
        for l in range(len(member_total_samples)//2):# total l round

            # Initialize global model
            self.attack_round = 0
            global_model = FederatedModel(self.reader, aggregator)
            global_model.init_global_model()
            test_loss, test_acc = global_model.test_outcome()

            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, "g", test_loss, test_acc, 0)
            logger.info("Global model initiated, loss={}, acc={}".format(test_loss, test_acc))

            # Initialize participants
            participants = []
            for i in range(NUMBER_OF_PARTICIPANTS):
                participants.append(FederatedModel(self.reader, aggregator))
                participants[i].init_participant(global_model, i)
                test_loss, test_acc = participants[i].test_outcome()
                try:
                    if DEFAULT_AGR == FANG or FL_TRUST:
                        aggregator.agr_model_acquire(global_model)
                except NameError:
                    pass

                # # Recording and printing
                logger.info("Participant {} initiated, loss={}, acc={}".format(i, test_loss, test_acc))

            # Initialize attacker
            attacker_success_round = []
            attacker = WhiteBoxMalicious(self.reader, aggregator)
            attacker.get_samples(member_total_samples[2*l:2*l+2]) # Get 2 samples for each round to attack
            logger.info(member_total_samples[2*l:2*l+2]) # Print the 2 samples
            attacker_sample_list,member_list,nonmember_list = attacker.attacker_sample()
            logger.info(member_list)
            for i in range(30):
                exec("honest_node_side_list%s=[]" % i)
            vector_hist = []
            honest_node_side_list=[]
            participant_0_hist = []
            loss_set = []
            removed_samples = []
            ascent_factor = ASCENT_FACTOR
            for i in range(30):
                exec("correct_set%s = {}" % i)
            for i in range(30):
                exec("correct_set_dic%s = {}" % i)
            for j in range(MAX_EPOCH):

                # The global model's parameter is shared to each participant before every communication round
                global_parameters = global_model.get_flatten_parameters()
                train_acc_collector = []
                for i in range(NUMBER_OF_PARTICIPANTS):
                    
                    # The participants collect the global parameters before training
                    participants[i].collect_parameters(global_parameters)
                    if i == 0:
                        current_hist, attack_ground_0, label_status, out_list = participants[i].check_member_label(member_list)
                        participant_0_hist.append(current_hist)
                        vector_hist.append(out_list)
                        for index in range(len(out_list)):
                            logger.info("current epoch {}, attack sample {} prediction is {} groudtruth is {}".format(j,
                                                                                                                      out_list[
                                                                                                                          index][
                                                                                                                          3],
                                                                                                                      out_list[
                                                                                                                          index][
                                                                                                                          1],
                                                                                                                      out_list[
                                                                                                                          index][
                                                                                               2]))

                        honest_node_side,correct_set,correct_set_dic = participants[i].detect_node_side_vector(all_member_samples,last_batch)
                    else:
                        honest_node_side, correct_set, correct_set_dic= participants[i].normal_detection_vector(i)

                    exec("honest_node_side_list%s = honest_node_side" % i)
                    honest_node_side_list.append(honest_node_side)
                    remove_counter = 0

                    #start from 100 epoch
                    if j == 100: #record the prediction of the 100 epoch as the baseline
                        exec("correct_set%s = correct_set" % i)
                        print(eval("correct_set%s " % i))
                        exec("correct_set_dic%s = correct_set_dic" % i)
                        print(eval("correct_set_dic%s " % i))
                    if j>100: #start to detect the attack based on prediction
                        monitor_list = participants[i].detect_attack_vector(eval("correct_set%s " % i),
                                                                            eval("correct_set_dic%s" % i)) #detect the attack
                        if monitor_list != []:
                            for n in monitor_list:
                                removed_samples.append([j,i,int(n.item())])
                                participants[i].del_defence(n) #remove the sample from the dataset
                                remove_counter+=1
                                logger.info("remove sample {}, total remove {}".format(n.item(),remove_counter))
                    logger.info("remove sample in this round {}".format(remove_counter))

                    # The participants calculate local gradients and share to the aggregator
                    participants[i].share_gradient()
                    train_loss, train_acc = participants[i].train_loss, participants[i].train_acc
                    train_acc_collector.append(train_acc)

                    # Printing and recording
                    test_loss, test_acc = participants[i].test_outcome()
                    try:
                        if DEFAULT_AGR == FANG or FL_TRUST:
                            aggregator.agr_model_acquire(global_model)
                    except NameError:
                        pass

                    logger.info(
                        "Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i,
                                                                                                                 test_loss,
                                                                                                                 test_acc,
                                                                                                                 train_loss,
                                                                                                                 train_acc))
                # attacker attack
                attacker.collect_parameters(global_parameters)
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()

                if true_member and false_member and true_non_member and false_non_member != 0:
                    attack_precision = true_member / (true_member + false_member)
                    attack_accuracy = (true_member + true_non_member) / (
                            true_member + true_non_member + false_member + false_non_member)
                    attack_recall = true_member / (true_member + false_non_member)
                else:
                    attack_precision = (true_member + 1) / (true_member + false_member + 1)
                    attack_accuracy = (true_member + true_non_member + 1) / (
                            true_member + true_non_member + false_member + false_non_member + 1)
                    attack_recall = (true_member + 1) / (true_member + false_non_member + 1)



                # attacker normal training
                if j < TRAIN_EPOCH or self.attack_round > 40:
                    attacker.train()

                # attacker attack
                else:
                    logger.info("attack!!")
                    logger.info("attack {}".format(self.attack_round))
                    try:
                        attacker.optimized_gradient_ascent(ascent_factor=ascent_factor,cover_factor=COVER_FACTOR)
                    except NameError:
                        attacker.optimized_gradient_ascent(ascent_factor=ascent_factor)


                    self.attack_round += 1



                if DEFAULT_AGR in [FANG, MULTI_KRUM]:
                    logger.info("Selected inputs are from participants number{}".format(aggregator.robust.appearence_list))
                    if 30 in aggregator.robust.appearence_list:
                        attacker_success_round.append(j)

                    logger.info("current status {}".format(str(aggregator.robust.status_list)))

                logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision,
                                                                                     attack_recall))
                pred_acc_member = attacker.evaluate_member_accuracy()[0].cpu()
                member_prediction = attacker.evaluate_member_accuracy()[1]
                logger.info("member prediction {}".format(member_prediction))
                pred_acc_non_member = attacker.evaluate_non_member_accuracy().cpu()

                attack_recorder.loc[len(attack_recorder)] = (j + 1, \
                                                             attack_accuracy, attack_precision, attack_recall, \
                                                             pred_acc_member, pred_acc_non_member, \
                                                             true_member, false_member, true_non_member, false_non_member)
                # Global model collects the aggregated gradient
                global_model.apply_gradient()
                loss_set.append(aggregator.robust.status_list+[j])
                # Printing and recording
                test_loss, test_acc = global_model.test_outcome()
                train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
                self.last_acc = train_acc
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
                logger.info(
                    "Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc,
                                                                                            train_acc))

            

            logger.info("attack success round {}, total {}".format(attacker_success_round, len(attacker_success_round)))

            honest_node_predicted_vector_collector = pd.DataFrame(removed_samples)
            honest_node_predicted_vector_collector.to_csv(
                EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "predicted_vector" + "_round" + str(l) + str(
                    MAX_EPOCH - TRAIN_EPOCH) + "optimized_model_single_honest_side_vector_defence_removed.csv")
            for i in range(30):
                honest_node_predicted_vector_collector = pd.DataFrame(eval("honest_node_side_list%s"%i))
                honest_node_predicted_vector_collector.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                        "TrainEpoch" + str(TRAIN_EPOCH) + "predicted_vector" + "_round" +str(l)+str(
                        MAX_EPOCH - TRAIN_EPOCH) + "optimized_model_single_honest_side_vector_defence_%s.csv"%i)

            predicted_vector_collector = pd.DataFrame(vector_hist)
            predicted_vector_collector.to_csv(
                EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "predicted_vector" + "_round" + str(l) + str(
                    MAX_EPOCH - TRAIN_EPOCH) + "optimized_model_single_vector_defence.csv")

            if record_model:
                param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
                for i in range(NUMBER_OF_PARTICIPANTS):
                    param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
                param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models_honest_side_vector_defence.csv")
            if record_process:
                recorder_suffix = "honest_node"
                acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                    "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                    MAX_EPOCH - TRAIN_EPOCH) + recorder_suffix + "_round" +str(l)+"optimized_model_single_ascent_factor{}_honest_side_vector_defence.csv".format(ASCENT_FACTOR))
                attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                       "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                    MAX_EPOCH - TRAIN_EPOCH) + recorder_suffix + "_round" +str(l) +"optimized_attacker_single_ascent_factor{}_honest_side_vector_defence.csv".format(ASCENT_FACTOR))






logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY,
                     'log_{}_{}_{}_TrainEpoch{}_AttackEpoch{}_honest_member_rate{}_ascent_factor{}_single_honest_side_vector_defence'.format(TIME_STAMP, DEFAULT_SET, DEFAULT_AGR,
                                                                               TRAIN_EPOCH,
                                                                               MAX_EPOCH - TRAIN_EPOCH,BLACK_BOX_MEMBER_RATE,ASCENT_FACTOR))

org = Organizer()
org.set_random_seed()
org.honest_node_vector_defense(logger)