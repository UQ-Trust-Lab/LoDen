from models import *
from constants import *
import pandas as pd
import numpy as np
import copy, os, random
import matplotlib.pyplot as plt
from common import DEVICE
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


    def honest_test(self, logger, adaptive=False, record_process=True, record_model=False, plot=False):
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
        member_total_samples = self.reader.get_honest_node_member()
        member_total_samples = member_total_samples[2:100]

        # Initialize attacker
        logger.info(len(member_total_samples)//2)
        for l in range(len(member_total_samples)//2):
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
                logger.info("Participant {} initiated, loss={}, acc={}".format(i, test_loss, test_acc))

            attacker_success_round = []
            print(l)
            attacker = WhiteBoxMalicious(self.reader, aggregator)
            print(member_total_samples[0:2])
            attacker.get_samples(member_total_samples[2*l:2*l+2])
            logger.info(member_total_samples[2*l:2*l+2])
            attacker_sample_list,member_list,nonmember_list = attacker.attacker_sample()
            logger.info(member_list)
            vector_hist = []
            participant_0_hist = []
            participant_0_hist_nonmember = []
            grad_data = []
            loss_set = []
            participant_1_hist = []
            participant_1_hist_nonmember = []
            participant_2_hist = []
            participant_2_hist_nonmember = []
            participant_3_hist = []
            participant_3_hist_nonmember = []
            participant_4_hist = []
            participant_4_hist_nonmember = []
            monitor_recorder = {}
            ascent_factor = ASCENT_FACTOR
            for j in range(MAX_EPOCH):
                # The global model's parameter is shared to each participant before every communication round
                global_parameters = global_model.get_flatten_parameters()
                train_acc_collector = []
                for i in range(NUMBER_OF_PARTICIPANTS):
                    # The participants collect the global parameters before training
                    participants[i].collect_parameters(global_parameters)
                    if i == 0:
                        current_hist, attack_ground_0, label_status, out_list = participants[i].check_member_label(member_list)
                        current_nonmember_hist, nonmember_ground_0 = participants[i].check_nonmember_sample(i,
                                                                                                            attacker_sample_list)
                        participant_0_hist.append(current_hist)
                        participant_0_hist_nonmember.append(current_nonmember_hist)
                        vector_hist.append(out_list)
                        print(out_list)
                        # predicted_vector_collector.loc[len(predicted_vector_collector)] = (out_list)
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




                if j < TRAIN_EPOCH or self.attack_round > 40:
                    attacker.train()

                else:
                    logger.info("attack!!")
                    logger.info("attack {}".format(self.attack_round))
                    try:
                        attacker.optimized_gradient_ascent(ascent_factor=ascent_factor,cover_factor=COVER_FACTOR)
                    except NameError:
                        attacker.optimized_gradient_ascent(ascent_factor=ascent_factor)


                    self.attack_round += 1



                if DEFAULT_AGR in [FANG, MULTI_KRUM, KRUM]:
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

            self.print_log(0,participant_0_hist,participant_0_hist_nonmember,attack_ground_0, nonmember_ground_0)

            logger.info("attack success round {}, total {}".format(attacker_success_round, len(attacker_success_round)))


            predicted_vector_collector = pd.DataFrame(vector_hist)
            predicted_vector_collector.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                    "TrainEpoch" + str(TRAIN_EPOCH) + "predicted_vector" + "_round" +str(l)+str(
                    MAX_EPOCH - TRAIN_EPOCH) + "optimized_model_single.csv")

            if record_model:
                param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
                for i in range(NUMBER_OF_PARTICIPANTS):
                    param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
                param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models.csv")
            if record_process:
                recorder_suffix = "honest_node"
                acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                    "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                    MAX_EPOCH - TRAIN_EPOCH) + recorder_suffix + "_round" +str(l)+"optimized_model_single_ascent_factor{}.csv".format(ASCENT_FACTOR))
                attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                       "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                    MAX_EPOCH - TRAIN_EPOCH) + recorder_suffix + "_round" +str(l) +"optimized_attacker_single_ascent_factor{}.csv".format(ASCENT_FACTOR))
            if plot:
                self.plot_attack_performance(attack_recorder)





logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY,
                     'log_{}_{}_{}_TrainEpoch{}_AttackEpoch{}_honest_member_rate{}_ascent_factor{}_single'.format(TIME_STAMP, DEFAULT_SET, DEFAULT_AGR,
                                                                               TRAIN_EPOCH,
                                                                               MAX_EPOCH - TRAIN_EPOCH,BLACK_BOX_MEMBER_RATE,ASCENT_FACTOR))

org = Organizer()
org.set_random_seed()
org.honest_test(logger)
