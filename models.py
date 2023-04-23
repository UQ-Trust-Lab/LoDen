import logging
import os
import sys
from data_reader import DataReader
from aggregator import *

def make_logger(name, save_dir, save_filename):
    """
    Make a logger to record the training process
    :param name: logger name
    :param save_dir: the directory to save the log file
    :param save_filename: the filename to save the log file
    :return: logger
    """
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt=DATE_FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, save_filename + ".txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def select_by_threshold(to_share: torch.Tensor, select_fraction: float, select_threshold: float = 1):
    """
    Apply the privacy-preserving method following selection-by-threshold approach
    :param to_share: the tensor to share
    :param select_fraction: the fraction of the tensor to share
    :param select_threshold: the threshold to select the tensor
    :return: the shared tensor and the indices of the selected tensor
    """
    threshold_count = round(to_share.size(0) * select_threshold)
    selection_count = round(to_share.size(0) * select_fraction)
    indices = to_share.topk(threshold_count).indices
    perm = torch.randperm(threshold_count).to(DEVICE)
    indices = indices[perm[:selection_count]]
    rei = torch.zeros(to_share.size()).to(DEVICE)
    rei[indices] = to_share[indices].to(DEVICE)
    to_share = rei.to(DEVICE)
    return to_share, indices


class ModelLocation30(torch.nn.Module):
    """
    The model to handel Location100 dataset
    """

    def __init__(self):
        super(ModelLocation30, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(446, 512),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 30),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class ModelPurchase100(torch.nn.Module):
    """
    The model handling purchase-100 data set
    """

    def __init__(self):
        super(ModelPurchase100, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(600, 1024),
            torch.nn.ReLU()
        )
        self.hidden_layer = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(256, 100)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        out = self.output_layer(out)
        return out


class ModelPreTrainedCIFAR10(torch.nn.Module):
    """
    The model to support pre-trained CIFAR-10 data set
    """

    def __init__(self):
        super(ModelPreTrainedCIFAR10, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(64, 1024),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class ModelLocation30(torch.nn.Module):
    """
    The model to handel Location100 dataset
    """

    def __init__(self):
        super(ModelLocation30, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(446, 512),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 30),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class ModelTexas100(torch.nn.Module):
    """
    The model to handel Texas10 dataset
    """

    def __init__(self):
        super(ModelTexas100, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(6169, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 100)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class TargetModel:
    """
    The model to attack against, the target for attacking
    """

    def __init__(self, data_reader: DataReader, participant_index=0, model=DEFAULT_SET):
        # initialize the model
        if model == PURCHASE100:
            self.model = ModelPurchase100()
        elif model == CIFAR_10:
            self.model = ModelPreTrainedCIFAR10()
        elif model == LOCATION30:
            self.model = ModelLocation30()
        elif model == TEXAS100:
            self.model = ModelTexas100()
        else:
            raise NotImplementedError("Model not supported")

        self.model = self.model.to(DEVICE)

        # initialize the data
        self.test_set = None
        self.train_set = None
        self.last_train_batch = None
        ## for 1-30 clients
        for i in range(1,30):
            exec("self.train_set%s=None" % i)
            exec("self.train_set_last_batch%s=None" % i)

        self.data_reader = data_reader
        self.participant_index = participant_index
        self.load_data()


        # initialize the loss function and optimizer
        self.loss_function = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Initialize recorder
        self.train_loss = -1
        self.train_acc = -1

        # Initialize confidence recorder
        self.mask = torch.ones(BATCH_SIZE)
        self.defend = False
        self.defend_count_down = 0
        self.defend_loss_checker = self.train_loss
        self.drop_out = BATCH_SIZE // 4


    def load_last_batch(self):
        """
        Load the last batch of the data reader
        """
        self.last_train_batch = self.data_reader.get_last_train_batch().to(DEVICE)
        self.last_test_batch = self.data_reader.get_last_test_batch().to(DEVICE)

    def load_data(self):
        """
        Load batch indices from the data reader
        :return: None
        """

        self.train_set= self.data_reader.get_train_set(self.participant_index).to(DEVICE)
        self.test_set = self.data_reader.get_test_set(self.participant_index).to(DEVICE)



    def normal_epoch(self, print_progress=False, by_batch=BATCH_TRAINING):
        """
        Train a normal epoch with the given dataset
        :param print_progress: if print the training progress or not
        :param by_batch: True to train by batch, False otherwise
        :return: the training accuracy and the training loss value
        """
        train_loss = 0
        train_acc = 0
        batch_counter = 0
        if by_batch:
            for batch_indices in self.train_set:
                batch_counter += 1
                if print_progress and batch_counter % 100 == 0:
                    print("Currently training for batch {}, overall {} batches"
                          .format(batch_counter, self.train_set.size(0)))
                if self.defend:
                    batch_indices = batch_indices[self.mask == 1]
                batch_x, batch_y = self.data_reader.get_batch(batch_indices.type(torch.int64))
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                out = self.model(batch_x).to(DEVICE)
                batch_loss = self.loss_function(out, batch_y)
                train_loss += batch_loss.item()
                prediction = torch.max(out, 1).indices.to(DEVICE)
                batch_acc = (prediction == batch_y).sum().to(DEVICE)
                train_acc += batch_acc.item()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            if self.last_train_batch !=None and len(self.last_train_batch) != 0:

                batch_x, batch_y = self.data_reader.get_batch(self.last_train_batch.type(torch.int64))
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                out = self.model(batch_x).to(DEVICE)
                batch_loss = self.loss_function(out, batch_y)
                train_loss += batch_loss.item()
                prediction = torch.max(out, 1).indices.to(DEVICE)
                batch_acc = (prediction == batch_y).sum().to(DEVICE)
                train_acc += batch_acc.item()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
        else:
            batch_x, batch_y = self.data_reader.get_batch(self.train_set)
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            out = self.model(batch_x[:10]).to(DEVICE)
            batch_loss = self.loss_function(out, batch_y)
            train_loss += batch_loss.item()
            prediction = torch.max(out, 1).indices.to(DEVICE)
            batch_acc = (prediction == batch_y).sum()
            train_acc += batch_acc.item()
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        if self.last_train_batch!=None:
            self.train_acc = train_acc / ((self.train_set.flatten().size(0)+self.last_train_batch.flatten().size(0)))
            self.train_loss = train_loss / ((self.train_set.flatten().size(0)+self.last_train_batch.flatten().size(0)))
        else:
            self.train_acc = train_acc / (self.train_set.flatten().size(0))
            self.train_loss = train_loss / (self.train_set.flatten().size(0))
        if print_progress:
            print("Epoch complete for participant {}, train acc = {}, train loss = {}"
                  .format(self.participant_index, train_acc, train_loss))
        return self.train_loss, self.train_acc

    def test_outcome(self, by_batch=BATCH_TRAINING):
        """
        Test through the test set to get loss value and accuracy
        :return: the test accuracy and test loss value
        """
        test_loss = 0
        test_acc = 0
        if by_batch:
            for batch_indices in self.test_set:
                batch_x, batch_y = self.data_reader.get_batch(batch_indices.type(torch.int64))
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                with torch.no_grad():
                    out = self.model(batch_x).to(DEVICE)
                    batch_loss = self.loss_function(out, batch_y).to(DEVICE)
                    test_loss += batch_loss.item()
                    prediction = torch.max(out, 1).indices.to(DEVICE)
                    batch_acc = (prediction == batch_y).sum().to(DEVICE)
                    test_acc += batch_acc.item()
        else:
            batch_x, batch_y = self.data_reader.get_batch(self.test_set.type(torch.int64))
            with torch.no_grad():
                out = self.model(batch_x)
                batch_loss = self.loss_function(out, batch_y)
                test_loss += batch_loss.item()
                prediction = torch.max(out, 1).indices
                batch_acc = (prediction == batch_y).sum()
                test_acc += batch_acc.item()
        test_acc = test_acc / (self.test_set.flatten().size(0))
        test_loss = test_loss / (self.test_set.flatten().size(0))
        return test_loss, test_acc

    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0).to(DEVICE)
        with torch.no_grad():
            for parameter in self.model.parameters():
                out = torch.cat([out, parameter.flatten()]).to(DEVICE)
        return out

    def load_parameters(self, parameters: torch.Tensor):
        """
        Load parameters to the current model using the given flatten parameters
        :param parameters: The flatten parameter to load
        :return: None
        """
        start_index = 0
        for param in self.model.parameters():
            length = len(param.flatten())
            to_load = parameters[start_index: start_index + length].to(DEVICE)
            to_load = to_load.reshape(param.size()).to(DEVICE)
            with torch.no_grad():
                param.copy_(to_load).to(DEVICE)
            start_index += length

    def get_epoch_gradient(self, apply_gradient=True):
        """
        Get the gradient for the current epoch
        :param apply_gradient: if apply the gradient or not
        :return: the tensor contains the gradient
        """
        cache = self.get_flatten_parameters().to(DEVICE)
        self.normal_epoch()
        gradient = self.get_flatten_parameters() - cache.to(DEVICE)
        if not apply_gradient:
            self.load_parameters(cache)
        return gradient

    def init_parameters(self, mode=INIT_MODE):
        """
        Initialize the parameters according to given mode
        :param mode: the mode to init with
        :return: None
        """
        if mode == NORMAL:
            to_load = torch.randn(self.get_flatten_parameters().size())
            self.load_parameters(to_load)
        elif mode == UNIFORM:
            to_load = torch.rand(self.get_flatten_parameters().size())
            self.load_parameters(to_load)
        elif mode == PYTORCH_INIT:
            return
        else:
            raise ValueError("Invalid initialization mode")

    def test_gradients(self, gradient: torch.Tensor):
        """
        Make use of the given gradients to run a test, then revert back to the previous status
        """
        cache = self.get_flatten_parameters().to(DEVICE)
        test_param = cache + gradient.to(DEVICE)
        self.load_parameters(test_param)
        loss, acc = self.test_outcome()
        self.load_parameters(cache)
        return loss, acc

    def get_gradzero(self, revert = True):
        """
        Get the gradient of the current model with zero gradient for aggregation usage
        :param revert: if revert the model to the previous status
        :return: the gradient of the current model
        """
        validation_data,validation_label = self.data_reader.get_batch(self.data_reader.fl_trust.type(torch.int64))
        cache = self.get_flatten_parameters()
        out = self.model(validation_data)
        loss = self.loss_function(out, validation_label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        if revert:
            self.load_parameters(cache)
        return gradient


class FederatedModel(TargetModel):
    """
    Representing the class of federated learning members
    """
    def __init__(self, reader: DataReader, aggregator: Aggregator, participant_index=0):
        super(FederatedModel, self).__init__(reader, participant_index)
        self.aggregator = aggregator
        self.member_list=[]
        self.nonmember_list = []

    def update_aggregator(self,aggregator):
        """
        Update the aggregator of the current model
        :param aggregator: the aggregator to update
        :return: None
        """
        self.aggregator = aggregator.to(DEVICE)

    def get_aggregator(self):
        """
        Get the aggregator of the current model
        :return: the aggregator of the current model
        """
        return self.aggregator

    def init_global_model(self):
        """
        Initialize the current model as the global model
        :return: None
        """
        self.init_parameters()
        self.test_set = self.data_reader.test_set.to(DEVICE)
        self.train_set = None

    def init_participant(self, global_model: TargetModel, participant_index):
        """
        Initialize the current model as a participant
        :return: None
        """
        self.participant_index = participant_index
        self.load_parameters(global_model.get_flatten_parameters())
        self.load_data()

    def share_gradient(self):
        """
        Participants share gradient to the aggregator
        :return: None
        """
        gradient = self.get_epoch_gradient()
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices=indices, source=self.participant_index)
        return gradient


    def apply_gradient(self):
        """
        Global model applies the gradient
        :return: None
        """
        parameters = self.get_flatten_parameters().to(DEVICE)
        parameters += self.aggregator.get_outcome(reset=True).to(DEVICE)
        self.load_parameters(parameters)

    def collect_parameters(self, parameter: torch.Tensor):
        """
        Participants collect parameters from the global model
        :param parameter: the parameters shared by the global model
        :return: None
        """
        to_load = self.get_flatten_parameters().to(DEVICE)
        parameter, indices = select_by_threshold(parameter, PARAMETER_EXCHANGE_RATE, PARAMETER_SAMPLE_THRESHOLD)
        to_load[indices] = parameter[indices]
        self.load_parameters(to_load)

    def check_member(self,dataindex,participant_ind):
        """
        Check if the given data index is in the training set of the given participant
        :param dataindex: the data index to check
        :param participant_ind: the participant index to check
        :return: True if the given data index is in the training set of the given participant, False otherwise
        """
        result = None
        if dataindex in self.data_reader.get_train_set(participant_ind):
            result = True
        return result

    def detect_node_side(self,member,last_batch):
        """
        Detect the node side of the given data indices
        :param member: the data indices to detect
        :param last_batch: the last batch of data indices to detect
        :return: None
        """
        out_list = []
        correct_set = []
        for batch in member:
            for i in batch:
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                if prediction == sample_y and i.item() not in correct_set:
                    correct_set.append(i.item())
                out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])
        for i in last_batch:
            sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
            out = self.model(sample_x)
            prediction = torch.max(out, -1).indices.to(DEVICE)
            probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
            if prediction == sample_y and i.item() not in correct_set:
                correct_set.append(i.item())
            out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])
        return out_list,correct_set


    def detect_node_side_vector(self,member,last_batch):
        """
        Detect the node side of the given data indices by probability vector
        :param member: the data indices to detect
        :param last_batch: the last batch of data indices to detect
        :return: None
        """
        correct_set_dic = {}
        correct_set = []
        out_list = []
        for batch in member:
            for i in batch:
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                if prediction == sample_y and i.item() not in correct_set_dic.keys():
                    correct_set_dic[i.item()] = float(probs[sample_y])
                    correct_set.append(i.item())
                out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])
        for i in last_batch:
            sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
            out = self.model(sample_x)
            prediction = torch.max(out, -1).indices.to(DEVICE)
            probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
            if prediction == sample_y and i.item() not in correct_set_dic.keys():
                correct_set_dic[i.item()] = float(probs[sample_y])
                correct_set.append(i.item())
            out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])
        return out_list,correct_set,correct_set_dic

    def check_member_label(self,member):
        """
        Check the label of the given data indices
        :param member: the data indices to check
        :return: None
        """
        attacker_ground = []
        pred_label = {}
        label_flag = []
        out_list = []
        for i in member:
            sample_x,sample_y = self.data_reader.get_batch(i.type(torch.int64))
            out = self.model(sample_x)
            prediction = torch.max(out, -1).indices.to(DEVICE)
            probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
            out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])

        return pred_label,attacker_ground,label_flag,out_list

    def check_label_on_samples(self,participant_ind,attacker_samples):
        """
        Check the label of the given data indices
        :param participant_ind: the participant index to check
        :param attacker_samples: the data indices to check
        :return: None
        """
        for sample_ind in range(len(attacker_samples)):
            if attacker_samples[sample_ind] in self.data_reader.train_set and sample_ind not in [x[0] for x in self.member_list]:
                self.member_list.append((sample_ind, attacker_samples[sample_ind]))
        attack_x,attack_y = self.data_reader.get_batch(attacker_samples.type(torch.int64)).to(DEVICE)
        sample_x,sample_y = self.data_reader.get_batch(self.data_reader.train_set[participant_ind].type(torch.int64)).to(DEVICE)
        attacker_ground = []
        pred_label = {}
        label_flag = []
        out_list = []
        for i in self.member_list:
            out = self.model(attack_x[i[0]])
            prediction = torch.max(out,-1).indices.to(DEVICE)
            probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
            attack_label = attack_y[i[0]]
            out_list.append([float(probs[attack_label]),int(prediction),int(attack_label),i[1].item()])
            if prediction == attack_label:
                label_flag.append("same {}, ground {}".format(i,int(attack_label)))
            else:
                label_flag.append("different {}, ground {}, predicted label {} ".format(i,int(attack_label),int(prediction)))
            pred_label[i[1]]=int(prediction)
            attacker_ground.append((int(i[1]),int(attack_label)))
        return pred_label,attacker_ground,label_flag,out_list

    def check_nonmember_sample(self,participant_ind,attacker_samples):
        """
        Check the label of the given data indices
        :param participant_ind: the participant index to check
        :param attacker_samples: the data indices to check
        :return: None
        """
        nonmembers_x,nonmembers_y = self.data_reader.get_batch(attacker_samples[2:].type(torch.int64))
        nonmember_ground = []
        pred_label_nonmember = {}
        for i in range(len(attacker_samples[2:])):
            out = self.model(nonmembers_x[i]).to(DEVICE)
            prediction = torch.max(out,-1).indices.to(DEVICE)
            nonmember_label = nonmembers_y[i]
            pred_label_nonmember[int(attacker_samples[2:][i])]=int(prediction)
            nonmember_ground.append((int(attacker_samples[2:][i]),int(nonmember_label)))

        return pred_label_nonmember,nonmember_ground

    def detect_attack(self,participant_ind):
        """
        Detect the attack vector
        :param participant_ind: the participant index to check
        :return: None
        """
        targeted_samples_monitor = []
        for batch_num, batch in enumerate(self.train_set):
            for num, i in enumerate(batch):
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                if prediction!=sample_y:
                    targeted_samples_monitor.append(i)
        return targeted_samples_monitor
   
    def detect_attack_vector(self,correct_set,correct_set_dic):
        """
        Detect the attack vector
        :param participant_ind: the participant index to check
        :return: None
        """
        targeted_samples_monitor = []
        for batch_num, batch in enumerate(self.train_set):
            for num, i in enumerate(batch):
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                if i.item() in correct_set and correct_set_dic[i.item()]-float(probs[sample_y]) >0.2:
                    targeted_samples_monitor.append(i)
                elif i.item() in correct_set and prediction!=sample_y:
                    targeted_samples_monitor.append(i)
        return targeted_samples_monitor
   
    def normal_detection(self,participant_ind):
        """
        Detect the attack vector
        :param participant_ind: the participant index to check
        :return: None
        """
        out_list = []
        correct_set = []
        for batch_num, batch in enumerate(self.data_reader.get_train_set(participant_ind)):
            for num, i in enumerate(batch):
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                if prediction == sample_y and i.item() not in correct_set:
                    correct_set.append(i.item())
                out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])

        return out_list,correct_set
    
    def normal_detection_vector(self,participant_ind):
        """
        Detect the attack vector
        :param participant_ind: the participant index to check
        :return: None
        """
        out_list = []
        correct_set_dic = {}
        correct_set =[]
        for batch_num, batch in enumerate(self.data_reader.get_train_set(participant_ind)):
            for num, i in enumerate(batch):
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                if prediction == sample_y and i.item() not in correct_set_dic.keys():
                    correct_set_dic[i.item()] = float(probs[sample_y])
                    correct_set.append(i.item())
                out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])

        return out_list,correct_set,correct_set_dic
    
    def del_defence(self, position):
        """
        Delete the samples from the train set
        :param position: the position of the samples to delete
        :return: None
        """
        self.train_set,self.last_train_batch = self.data_reader.del_samples(position, self.train_set,self.last_train_batch)




class WhiteBoxMalicious(FederatedModel):
    """
    Representing the malicious participant trying to collect data for a white-box membership inference attack
    """

    def __init__(self, reader: DataReader, aggregator: Aggregator):
        super(WhiteBoxMalicious, self).__init__(reader, aggregator, 0)
        self.members = None
        self.non_members = None
        self.batch_x = None
        self.batch_y = None
        self.rest = None
        try:
            if DEFAULT_AGR == FANG or DEFAULT_AGR == FL_TRUST:
                self.attack_samples,self.members, self.non_members = reader.get_black_box_batch_fixed_balance_class()
            else:
                self.attack_samples, self.members, self.non_members = reader.get_black_box_batch_fixed()
        except NameError:
                self.attack_samples, self.members, self.non_members = reader.get_black_box_batch_fixed()

        self.rest = self.non_members
        self.descending_samples = None
        self.shuffled_labels = {}
        self.shuffle_labels()

        self.global_gradient = torch.zeros(self.get_flatten_parameters().size())
        self.last_round_shared_grad = None
        self.pred_history = []
        self.pred_history.append([])
        self.pred_history.append([])
        self.pred_history_new = {}
        self.confidence_history = []
        self.confidence_history.append([])
        self.confidence_history.append([])
        self.member_prediction = None
        self.member_intersections = {}
        self.sample_hist ={}


    def train(self ,mislead_factor = 3,norm_scalling = 1, attack=False, mislead=True,
              ascent_factor=ASCENT_FACTOR, ascent_fraction=FRACTION_OF_ASCENDING_SAMPLES,white_box_optimize=False):
        """
        Start a white-box training
        """
        self.record_targeted_samples()
        norm_scalling =NORM_SCALLING
        print("train !!")
        gradient = self.gradient_ascent(ascent_factor=ascent_factor, adaptive_factor=ascent_fraction, mislead=mislead,mislead_factor=mislead_factor)
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        if attack:
            random_key = torch.randint(1, 10, [1]).item()
            if self.global_gradient is not None and white_box_optimize and random_key <8:
                norm = self.global_gradient.norm()
                gradient += self.global_gradient
                gradient = gradient * norm *norm_scalling / gradient.norm()
            self.last_round_shared_grad = gradient
        self.aggregator.collect(gradient, indices)
        return gradient

    def gradient_ascent(self, ascent_factor=ASCENT_FACTOR, batch_size=BATCH_SIZE,
                        adaptive_factor=FRACTION_OF_ASCENDING_SAMPLES, mislead=False, mislead_factor=1):
        """
        Take one step of gradient ascent, the returned gradient is a combination of ascending gradient, descending
        gradient, and misleading gradient
        :return: gradient generated
        """
        cache = self.get_flatten_parameters()
        threshold = round(len(self.attack_samples) * adaptive_factor)

        # Perform gradient ascent for ascending samples
        if RESERVED_SAMPLE != 0:
            cover_samples = self.data_reader.reserve_set
            i = 0
            while i * batch_size < len(cover_samples):
                batch_index = cover_samples[i * batch_size:(i + 1) * batch_size]
                x, y = self.data_reader.get_batch(batch_index.type(torch.int64))
                out = self.model(x)
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
            cov_gradient = self.get_flatten_parameters() - cache
            self.load_parameters(cache)
        # Perform gradient descent for the rest of samples
        i = 0
        while i * batch_size < len(self.attack_samples):
            batch_index =  self.attack_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index.type(torch.int64))
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        desc_gradient = self.get_flatten_parameters() - cache
        if RESERVED_SAMPLE != 0:
            final_gradient = desc_gradient+cov_gradient
        else:
            final_gradient = desc_gradient
        return final_gradient

    def attacker_sample(self) :
        """
        Get the attack samples
        :return: The attack samples
        """
        return self.attack_samples, self.members, self.non_members

    def get_samples(self,members):
        """
        Get the samples for the given members
        :param members: The members to get the samples for
        :return: None
        """
        self.members = members
        self.batch_x, self.batch_y = self.data_reader.get_batch(self.members.type(torch.int64))


    def target_participants(self,participant_index):
        """
        Get the target participants
        :param participant_index: The participant index to get the target participants for
        :return: None
        """
        self.attack_samples_fixed = self.attack_samples_fixed[:int((NUMBER_OF_ATTACK_SAMPLES * BLACK_BOX_MEMBER_RATE / NUMBER_OF_PARTICIPANTS) * (participant_index + 1))]


    def optimized_gradient_ascent(self, batch_size=BATCH_SIZE, ascent_factor=ASCENT_FACTOR,
                         mislead=False, mislead_factor=1,cover_factor = 2):
        """
        Take one step of gradient ascent, the returned gradient is a combination of ascending gradient, descending
        gradient, and misleading gradient
        :return: gradient generated
        """
        print("ascent_factor {}, cover factor {}".format(ascent_factor,cover_factor))
        cache = self.get_flatten_parameters()
        self.load_parameters(cache)
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        ascent_gradient = - ascent_factor * gradient
        if RESERVED_SAMPLE != 0:
            self.load_parameters(cache)
            cover_samples = self.data_reader.reserve_set
            i = 0
            while i * batch_size < len(cover_samples):
                batch_index = cover_samples[i * batch_size:(i + 1)]
                x, y = self.data_reader.get_batch(batch_index.type(torch.int64))
                out = self.model(x)
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
            cov_gradient = cover_factor * self.get_flatten_parameters() - cache

        self.load_parameters(cache)
        x_rest, y_rest = self.data_reader.get_batch(self.rest.type(torch.int64))
        out = self.model(x_rest)
        loss = self.loss_function(out, y_rest)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        normal_gradient = self.get_flatten_parameters() - cache
        if RESERVED_SAMPLE != 0:
            final_gradient = cov_gradient + normal_gradient + ascent_gradient
        else:
            final_gradient = cover_factor * normal_gradient + ascent_gradient 
        gradient, indices = select_by_threshold(final_gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices)

        return gradient



    def shuffle_labels(self, iteration=WHITE_BOX_SHUFFLE_COPIES):
        """
        Shuffle the labels in several random permutation, to be used as misleading labels
        it will repeat the given iteration times denote as k, k different copies will be saved
        :param iteration: The number of copies to be saved
        :return: None
        """
        max_label = torch.max(self.data_reader.labels).item()
        for i in range(iteration):
            shuffled = self.data_reader.labels[torch.randperm(len(self.data_reader.labels))]
            for j in torch.nonzero(shuffled == self.data_reader.labels):
                shuffled[j] = (shuffled[j] + torch.randint(max_label, [1]).item()) % max_label
            self.shuffled_labels[i] = shuffled


    def collect_parameters(self, parameter: torch.Tensor):
        """
        Save the parameters from last round before collect new parameters
        """
        cache = self.get_flatten_parameters()
        super(WhiteBoxMalicious, self).collect_parameters(parameter)
        self.global_gradient = self.get_flatten_parameters() - cache


    def evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        """
        result = []
        batch_x, batch_y = self.data_reader.get_batch(self.members.type(torch.int64))
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices.to(DEVICE)
        accurate = (prediction == batch_y).sum().to(DEVICE)
        for i in range(len(self.members)):
            out = self.model(batch_x[i])
            prediction = torch.max(out, -1).indices
            flag = (prediction == batch_y[i])
            result.append((self.members[i],batch_y[i]))
        return accurate/ len(batch_y),result

    def evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.non_members.type(torch.int64))
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices.to(DEVICE)
        accurate = (prediction == batch_y).sum().to(DEVICE)
        return accurate / len(batch_y)

    def evaluate_attack_result(self):
        """
        Evaluate the attack result, return the overall accuracy, member accuracy, and precise
        :return: the number of true member, false member, true non-member, false non-member
        """
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices

        for i in range(len(self.attack_samples)):
            if prediction[i] == batch_y[i]:
                attack_result.append(1)
            else:
                attack_result.append(0)
            if self.attack_samples[i] in self.data_reader.train_set:
                ground_truth.append(1)
            else:
                ground_truth.append(0)

            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            else:
                false_non_member += 1

        return true_member, false_member, true_non_member, false_non_member


    def optimized_evaluation_init(self):
        """
        Calculate the intersection of self.members and the train set of each participant
        """
        for i in range(NUMBER_OF_PARTICIPANTS):
            self.member_intersections[i] = \
                torch.tensor(np.intersect1d(self.data_reader.get_train_set(i).to(DEVICE), self.attack_samples.to(DEVICE)))

    def record_targeted_samples(self):
        for member in self.members:
            members_x, members_y = self.data_reader.get_batch(member.type(torch.int64))
            out = self.model(members_x)
            prediction = torch.max(out, -1).indices
            if prediction == members_y:
                result = 1
            else:
                result = 0
            self.sample_hist[int(member)]= int(self.sample_hist.get(int(member),0))+result


