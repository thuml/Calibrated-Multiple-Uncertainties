from data import *
from net import *
from lib import *
import datetime
from tqdm import tqdm

if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

log_dir = f'{args.log.root_dir}/{now}'

logger = SummaryWriter(log_dir)

with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc
}


class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)
        # self.discriminator_separate = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        d_0 = self.discriminator_separate(_)
        return y, d, d_0


totalNet = TotalNet()

# feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, device=device).train(
#     True)
# classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, device=device).train(True)
# # discriminator = nn.DataParallel(totalNet.discriminator, device_ids=gpu_ids, device=device).train(True)
# discriminator_separate = nn.DataParallel(totalNet.discriminator_separate, device_ids=gpu_ids,
#                                          device=device).train(True)
feature_extractor = totalNet.feature_extractor.to(device)
classifier = totalNet.classifier.to(device)
discriminator = totalNet.discriminator.to(device)
# discriminator_separate = totalNet.discriminator_separate.to(device)

if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    # discriminator.load_state_dict(data['discriminator'])
    # discriminator_separate.load_state_dict(data['discriminator_separate'])

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, \
            Accumulator(['feature', 'predict_prob', 'label', 'fc2_s',
                         'entropy', 'consistency']) as target_accumulator, \
            torch.no_grad():
        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            im = im.to(device)
            label = label.to(device)

            feature = feature_extractor.forward(im)
            feature, __, fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, predict_prob = classifier.forward(feature)
            # domain_prob = discriminator_separate.forward(__)

            entropy = get_entropy(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, domain_temperature=1.0,
                                  class_temperature=1.0)
            consistency = get_consistency(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5)

            # predict_prob = get_predict_prob(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5)

            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())

    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    entropy = normalize_weight(torch.tensor(entropy))
    consistency = normalize_weight(torch.tensor(consistency))
    target_share_weight = (entropy + consistency) / 2

    # for x in target_accumulator:
    # print(target_accumulator['target_share_weight'])
    # hist, bin_edges = np.histogram(target_share_weight, bins=20, range=(0, 1))
    # print(hist)
    # print(bin_edges)

    #
    # hist, bin_edges = np.histogram(consistency, bins=20, range=(0, 1))
    # print(hist)
    # print(bin_edges)

    ana = list(zip(entropy, consistency, label))
    array = sorted(ana, key=lambda x: x[0])
    np.savetxt("ana.csv", array, delimiter=',')


    # print(array)
    #
    # a1, a2, a3 = zip(*array)
    # print(a1)
    # print(a2)
    # print(a3)

    def outlier(each_target_share_weight, each_pred_id):
        return each_target_share_weight > args.test.w_0 or each_pred_id > 9


    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]

    for (each_predict_prob, each_label, each_target_share_weight) in zip(predict_prob, label, target_share_weight):
        if each_label in source_classes:
            counters[each_label].Ntotal += 1.0
            each_pred_id = np.argmax(each_predict_prob)
            if not outlier(each_target_share_weight, each_pred_id):
                counters[int(each_pred_id)].Npred += 1.0
            if not outlier(each_target_share_weight, each_pred_id) and each_pred_id == each_label:
                counters[each_label].Ncorrect += 1.0
        else:
            counters[-1].Ntotal += 1.0
            each_pred_id = np.argmax(each_predict_prob)
            if outlier(each_target_share_weight, each_pred_id):
                counters[-1].Ncorrect += 1.0
                counters[-1].Npred += 1.0

    class_ratio = [x.Npred for x in counters]
    print(class_ratio)

    acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
    print(acc_tests)
    acc_test = torch.ones(1, 1) * np.mean(acc_tests)
    print(f'test accuracy is {acc_test.item()}')
    exit(0)

# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay,
              momentum=args.train.momentum, nesterov=True), scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(classifier.bottleneck.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,
              momentum=args.train.momentum, nesterov=True), scheduler)
fc_para = [{"params": classifier.fc.parameters()}, {"params": classifier.fc2.parameters()},
           {"params": classifier.fc3.parameters()}, {"params": classifier.fc4.parameters()},
           {"params": classifier.fc5.parameters()}]
optimizer_fc = OptimWithSheduler(
    optim.SGD(fc_para, lr=args.train.lr * 5, weight_decay=args.train.weight_decay,
              momentum=args.train.momentum, nesterov=True), scheduler)
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,
              momentum=args.train.momentum, nesterov=True), scheduler)
# optimizer_discriminator_separate = OptimWithSheduler(
#     optim.SGD(discriminator_separate.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,
#               momentum=args.train.momentum, nesterov=True), scheduler)

global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step), desc='global step')
epoch_id = 0

while global_step < args.train.min_step:

    iters = tqdm(
        zip(source_train_dl, source_train_dl2, source_train_dl3, source_train_dl4, source_train_dl5, target_train_dl),
        desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
    epoch_id += 1

    for i, ((im_source, label_source), (im_source2, label_source2), (im_source3, label_source3),
            (im_source4, label_source4), (im_source5, label_source5), (im_target, label_target)) in enumerate(iters):

        feature_extractor.train()
        classifier.train()

        save_label_target = label_target  # for debug usage

        label_source = label_source.to(device)
        label_source2 = label_source2.to(device)
        label_source3 = label_source3.to(device)
        label_source4 = label_source4.to(device)
        label_source5 = label_source5.to(device)
        label_target = label_target.to(device)
        # label_target = torch.zeros_like(label_target)

        # =========================forward pass
        im_source = im_source.to(device)
        im_source2 = im_source2.to(device)
        im_source3 = im_source3.to(device)
        im_source4 = im_source4.to(device)
        im_source5 = im_source5.to(device)
        im_target = im_target.to(device)

        fc1_s = feature_extractor.forward(im_source)
        fc1_s2 = feature_extractor.forward(im_source2)
        fc1_s3 = feature_extractor.forward(im_source3)
        fc1_s4 = feature_extractor.forward(im_source4)
        fc1_s5 = feature_extractor.forward(im_source5)
        fc1_t = feature_extractor.forward(im_target)

        fc1_s, feature_source, fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, predict_prob_source = classifier.forward(fc1_s)
        fc1_s2, feature_source2, fc2_s_2, fc2_s2_2, fc2_s3_2, fc2_s4_2, fc2_s5_2, predict_prob_source2 = \
            classifier.forward(fc1_s2)
        fc1_s3, feature_source3, fc2_s_3, fc2_s2_3, fc2_s3_3, fc2_s4_3, fc2_s5_3, predict_prob_source3 = \
            classifier.forward(fc1_s3)
        fc1_s4, feature_source4, fc2_s_4, fc2_s2_4, fc2_s3_4, fc2_s4_4, fc2_s5_4, predict_prob_source4 = \
            classifier.forward(fc1_s4)
        fc1_s5, feature_source5, fc2_s_5, fc2_s2_5, fc2_s3_5, fc2_s4_5, fc2_s5_5, predict_prob_source5 = \
            classifier.forward(fc1_s5)
        fc1_t, feature_target, fc2_t, fc2_t2, fc2_t3, fc2_t4, fc2_t5, predict_prob_target = classifier.forward(fc1_t)

        # domain_prob_discriminator_source = discriminator.forward(feature_source)
        # domain_prob_discriminator_target = discriminator.forward(feature_target)

        # domain_prob_discriminator_source_separate = discriminator_separate.forward(feature_source.detach())
        # domain_prob_discriminator_target_separate = discriminator_separate.forward(feature_target.detach())

        # source_share_weight = get_source_share_weight(domain_prob_discriminator_source_separate, fc2_s,
        #                                               domain_temperature=1.0, class_temperature=10.0)
        # source_share_weight = normalize_weight(source_share_weight)
        # entropy = get_entropy(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5,
        #                       domain_temperature=1.0, class_temperature=1.0)
        # target_share_weight = normalize_weight(entropy)

        # source_share_weight = get_label_weight(label_source, common_classes)
        # target_share_weight = get_label_weight(label_target, common_classes)

        # ==============================compute loss
        # adv_loss = torch.zeros(1, 1).to(device)
        # adv_loss_separate = torch.zeros(1, 1).to(device)

        # tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source,
        #                                                          torch.ones_like(domain_prob_discriminator_source))
        # adv_loss += torch.mean(tmp, dim=0, keepdim=True)
        # tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target,
        #                                                          torch.zeros_like(domain_prob_discriminator_target))
        # adv_loss += torch.mean(tmp, dim=0, keepdim=True)

        # adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate,
        #                                   torch.ones_like(domain_prob_discriminator_source_separate))
        # adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate,
        #                                   torch.zeros_like(domain_prob_discriminator_target_separate))

        # ============================== cross entropy loss, it receives logits as its inputs
        # ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label_source)
        # ce = torch.mean(ce, dim=0, keepdim=True)

        ce = nn.CrossEntropyLoss()(fc2_s, label_source)
        ce2 = nn.CrossEntropyLoss()(fc2_s2_2, label_source2)
        ce3 = nn.CrossEntropyLoss()(fc2_s3_3, label_source3)
        ce4 = nn.CrossEntropyLoss()(fc2_s4_4, label_source4)
        ce5 = nn.CrossEntropyLoss()(fc2_s5_5, label_source5)

        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_fc, optimizer_discriminator]):
            # [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_discriminator_separate]):
            # loss = ce + adv_loss + adv_loss_separate
            loss = (ce + ce2 + ce3 + ce4 + ce5) / 5
            loss.backward()

        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))),
                                variable_to_numpy(predict_prob_source))
            acc_train = torch.tensor([counter.reportAccuracy()]).to(device)
            # logger.add_scalar('adv_loss', adv_loss, global_step)
            logger.add_scalar('ce', ce, global_step)
            # logger.add_scalar('adv_loss_separate', adv_loss_separate, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)

        if global_step % args.test.test_interval == 0:

            feature_extractor.eval()
            classifier.eval()
            entropy = None
            consistency = None

            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
            with TrainingModeManager([feature_extractor, classifier, discriminator], train=False) as mgr, \
                    Accumulator(['feature', 'predict_prob', 'label',
                                 'entropy', 'consistency']) as target_accumulator, \
                    torch.no_grad():

                for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                    im = im.to(device)
                    label = label.to(device)

                    feature = feature_extractor.forward(im)
                    feature, __, fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, predict_prob = classifier.forward(
                        feature)
                    # domain_prob = discriminator_separate.forward(__)

                    # target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                    #                                               class_temperature=1.0)
                    entropy = get_entropy(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5,
                                          domain_temperature=1.0, class_temperature=1.0).detach()
                    consistency = get_consistency(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5).detach()
                    # predict_prob = get_predict_prob(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5)

                    for name in target_accumulator.names:
                        globals()[name] = variable_to_numpy(globals()[name])

                    target_accumulator.updateData(globals())

            for x in target_accumulator:
                globals()[x] = target_accumulator[x]

            entropy = normalize_weight(torch.tensor(entropy))
            consistency = normalize_weight(torch.tensor(consistency))
            target_share_weight = (entropy + consistency) / 2
            print(target_share_weight.size())


            def outlier(each_target_share_weight):
                return each_target_share_weight > args.test.w_0


            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]

            for (each_predict_prob, each_label, each_target_share_weight) in zip(predict_prob, label,
                                                                                 target_share_weight):
                if each_label in source_classes:
                    counters[each_label].Ntotal += 1.0
                    each_pred_id = np.argmax(each_predict_prob)
                    if not outlier(each_target_share_weight) and each_pred_id == each_label:
                        counters[each_label].Ncorrect += 1.0
                else:
                    counters[-1].Ntotal += 1.0
                    if outlier(each_target_share_weight):
                        counters[-1].Ncorrect += 1.0

            acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
            print(acc_tests)
            acc_test = torch.ones(1, 1) * np.mean(acc_tests)

            logger.add_scalar('acc_test', acc_test, global_step)
            # clear_output()

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'discriminator': discriminator.state_dict() if not isinstance(discriminator, Nonsense) else 1.0,
                # 'discriminator_separate': discriminator_separate.state_dict(),
            }

            if acc_test > best_acc:
                best_acc = acc_test
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            with open(join(log_dir, 'sourceonly.pkl'), 'wb') as f:
                torch.save(data, f)
