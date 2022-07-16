# Parse input arguments
import argparse

from helpers.generic import str2bool

SUPPORTED_DATASETS = ['mnist-mnist', 'mnist-svhn', 'mnist-cdcb', 'mnist-multiply', 'mnist-add1',
                      'sketchy-vgg', 'cub', 'cub-imgft2sent']


class Options:
    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='Adversarially trained Wyner models')

        # general experiment setting
        parser.add_argument('--experiment', type=str, default='', metavar='E', help='experiment name')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--num-workers', type=int, default=2)

        # dataset
        parser.add_argument('--dataset', type=str, default="mnist-mnist", choices=SUPPORTED_DATASETS)
        parser.add_argument('--ssl', type=str2bool, default=False)
        parser.add_argument('--dc', type=int, default=10000,
                            help='maximum number of data points per class in paired dataset')
        parser.add_argument('--dm', type=int, default=1,
                            help='number of pairings for each example of synthetic datasets')

        # Sketchy data specific
        parser.add_argument('--sketchy-split', type=int, default=1, choices=[1, 2])
        parser.add_argument('--sketchy-retrieval-metric', type=str, default='cosine', choices=['euclidean', 'cosine'])
        parser.add_argument('--save-retrieved-images', action='store_true', default=False)
        parser.add_argument('--n-retrievals-to-save', type=int, default=10)
        parser.add_argument('--n-retrievals', type=int, default=100)

        # {Sketchy, CUB} data specific
        parser.add_argument('--instance-noise-scale-x', type=float, default=0.25)
        parser.add_argument('--instance-noise-scale-y', type=float, default=0.25)

        # CUB data specific
        #   CUB image (raw)
        parser.add_argument('--cub-imgsize', type=int, default=128)
        #   CUB image (feature)
        parser.add_argument('--cub-imgft-encoder-use-batchnorm', type=str2bool, default=True)
        parser.add_argument('--cub-imgft-decoder-use-batchnorm', type=str2bool, default=False)
        parser.add_argument('--cub-imgft-activation', type=str, default='lrelu', choices=['lrelu', 'elu'])
        parser.add_argument('--cub-imgft-scaling', type=str2bool, default=False)
        #   CUB sentence
        parser.add_argument('--cub-sent-normalize-embedding', action='store_true', default=False)
        parser.add_argument('--cub-sent-use-pretrained-embedding', type=str2bool, default=False)
        parser.add_argument('--cub-sent-trainable-embedding', type=str2bool, default=True)
        parser.add_argument('--cub-sent-embedding-dim', type=int, default=128)  # FIXME: currently only 128 is supported
        parser.add_argument('--cub-sent-use-conv2d', type=str2bool, default=True)
        parser.add_argument('--cub-sent-recon-fn', type=str, choices=['l1', 'l2', 'log', 'cos'], default='log')
        parser.add_argument('--cub-sent-adaptive-embedding-noise', type=str2bool, default=True)
        parser.add_argument('--cub-sent-tie-embedding', action='store_true', default=False,
                            help='The discriminator word embedding is identical to that of generator (experimental)')
        parser.add_argument('--cub-sent-tie-embedding-inverse', type=str2bool, default=True,
                            help='The weight of embedding inverse is tied to that of embedding (from MMVAE)')
        parser.add_argument('--cub-sent-use-pretrained-ae', type=str2bool, default=False)
        parser.add_argument('--cub-sent-output-embedding-during-training', action='store_true', default=False)

        # logging related
        parser.add_argument('--main-path', type=str, default="..",
                            help='main path where datasets live and loggings are saved')
        parser.add_argument('--silent', type=str2bool, default=True, help='mute tqdm verbosity')
        parser.add_argument('--overwrite', action='store_true')
        parser.add_argument('--temp', action='store_true', help='work on a temporary path')
        parser.add_argument('--evaluation-mode', choices=['all', 'j', 'c', 'cx2y', 'cy2x'], default='all')
        parser.add_argument('--ref-value-key', type=str, default=None,
                            choices=['err_j', 'err_cx2y', 'err_cy2x',
                                     'fd_j', 'fd_cx2y', 'fd_cy2x',
                                     'test_P@K', 'test_mAP@all',
                                     'cca_j', 'cca_cx2y', 'cca_cy2x', None])
        parser.add_argument('--save-optim', type=str2bool, default=False)

        # image sample visualization related
        parser.add_argument('--n-per-class-for-plot', type=int, default=3,
                            help='number of digits per class for conditional generation (if dataset in IMAGE_DATASETS)')
        parser.add_argument('--n-for-plot', type=int, default=15,
                            help='number of z for joint generation')
        parser.add_argument('--n-per-ref-for-plot', type=int, default=25,
                            help='number of different (u,v) with each z for joint/conditional generation')

        # evaluation related
        parser.add_argument("--call-eval-during-training", action='store_true')
        parser.add_argument("--call-eval-during-inference", action='store_true')
        parser.add_argument('--evaluation-freq', type=int, default=1, help='frequency for evaluation')
        parser.add_argument("--evaluate-only", action='store_true')
        parser.add_argument("--do-not-evaluate", action='store_true')
        parser.add_argument('--do-not-evaluate-fd-scores', action='store_true')
        parser.add_argument('--do-not-classify-samples', action='store_true')
        parser.add_argument('--do-not-classify-from-representations', action='store_true')
        parser.add_argument('--n-epochs-for-latent-classification', type=int, default=10)
        parser.add_argument('--latent-classification-freq', type=int, default=5)
        parser.add_argument('--do-not-save-samples', action='store_true')
        parser.add_argument('--do-not-test', action='store_true')
        parser.add_argument("--turn-off-additional-gaussian-noise-during-evaluation", action='store_true', default=False)
        parser.add_argument('--n-samples-for-joint-eval', type=int, default=16384)
        parser.add_argument('--n-per-class-for-cond-fd', type=int, default=32,
                            help='number of reference images per digits for conditional FD')
        parser.add_argument('--n-per-ref-for-cond-fd', type=int, default=2048,
                            help='number of different (z,u,v) per x/y for conditional FD')
        parser.add_argument('--fd-model-types', nargs=2, type=str,
                            default=['inception', 'inception'],
                            choices=['inception', 'mnist', 'svhn', 'mnist_ae', 'svhn_ae'])
        parser.add_argument('--fd-feature-dims', nargs=2, type=int, default=[2048, 2048])
        parser.add_argument("--output-filetype", default="jpg", choices=["png", "jpg"])
        parser.add_argument("--plot-reconstruction", type=str2bool, default=False)
        parser.add_argument("--plot-stochastic-reconstruction", type=str2bool, default=True)
        parser.add_argument("--path-to-saved-model", type=str, default='')

        # likelihood evaluation parameters (annealed importance sampling)
        parser.add_argument('--do-evaluate-likelihoods', action='store_true')
        parser.add_argument('--chain-length', type=int, default=10000)
        parser.add_argument('--n-batches-for-ais', type=int, default=16)
        parser.add_argument('--likelihood', type=str, default='normal', choices=['normal', 'bernoulli'])
        parser.add_argument('--likelihood-var', type=float, default=0.075,
                            help='variance for Gaussian observation model')
        parser.add_argument('--posterior-var', type=float, default=0.075, help='variance for Gaussian encoder')

        # model hyperparameters
        #   latent dimensions
        parser.add_argument('--dim-z', type=int, default=16, metavar='DZ', help='latent dimensionality (default: 16)')
        parser.add_argument('--dim-u', type=int, default=8, metavar='DU', help='latent dimensionality (default: 8)')
        parser.add_argument('--dim-v', type=int, default=8, metavar='DV', help='latent dimensionality (default: 8)')
        #   general setting
        parser.add_argument('--n-resblocks', type=int, default=0)
        parser.add_argument('--base-cc', type=int, default=32)
        parser.add_argument('--kernel-size', type=int, default=5, choices=[4, 5])
        parser.add_argument('--feature-map-last-activation', type=str2bool, default=True)
        #   network normalization
        parser.add_argument('--batchnorm-momentum', type=float, default=1e-3)
        parser.add_argument('--batchnorm-eps', type=float, default=1e-3)
        parser.add_argument('--generator-feature-use-batchnorm', type=str2bool, default=True,
                            help='use batch normalization in generator feature map')
        parser.add_argument('--generator-core-use-batchnorm', type=str2bool, default=True,
                            help='use batch normalization in core generator')
        parser.add_argument('--discriminator-feature-use-batchnorm', type=str2bool, default=True,
                            help='use batch normalization in discriminator feature map')
        parser.add_argument('--discriminator-feature-use-batchnorm-first-layer', type=str2bool, default=False,
                            help='use batch normalization in discriminator feature map first layer')
        parser.add_argument('--discriminator-core-use-batchnorm', type=str2bool, default=False,
                            help='use batch normalization in core discriminator')
        parser.add_argument('--discriminator-feature-use-spectralnorm', type=str2bool, default=False,
                            help='use spectral normalization in discriminator feature map')
        #   discriminator network hyperparameters
        parser.add_argument('--discriminator-n-hidden-layers', type=int, default=1)
        parser.add_argument('--discriminator-hidden-dim', type=int, default=512)
        parser.add_argument('--discriminator-negative-slope', type=float, default=0.2)
        #   encoder network hyperparameters
        parser.add_argument('--encoder-n-hidden-layers', type=int, default=0)
        parser.add_argument('--encoder-hidden-dim', type=int, default=0)
        parser.add_argument('--encoder-negative-slope', type=float, default=0.2)
        parser.add_argument('--prior-type', type=str, default='gaussian', choices=['gaussian', 'uniform'])
        #   decoder network hyperparameters
        parser.add_argument('--decoder-negative-slope', type=float, default=0.2)
        #   degenerate modes
        parser.add_argument('--implicit', action='store_true', default=False,
                            help='hide u, v in the objective functions')
        parser.add_argument('--marginalize-bottleneck', action='store_true', default=False,
                            help='hide z in the objective functions (a.k.a. CVAE)')
        parser.add_argument('--degenerate-local-encoder', action='store_true', default=False,
                            help='ignore conditioning with z in local encoders (a.k.a. VCCA-private)')
        parser.add_argument('--simple-local-encoder', type=str2bool, default=True,
                            help='use a linear layer to infer local variables')
        #   encoder randomness
        parser.add_argument('--joint-encoder-noise-dim', type=int, default=0, metavar='DNE',
                            help='dimensionality of joint encoder noise (default: 0)')
        parser.add_argument('--joint-encoder-add-noise-channel', action='store_true',
                            help='whether to add noise channel in joint encoder')
        parser.add_argument('--joint-encoder-additional-noise-sigma', type=float, default=0.0, metavar='SNE',
                            help='sigma for additional Gaussian noise at joint encoder')
        parser.add_argument('--marginal-encoder-noise-dim', type=int, default=0, metavar='DNE',
                            help='dimensionality of marginal encoder noise (default: 0)')
        parser.add_argument('--marginal-encoder-add-noise-channel', action='store_true',
                            help='whether to add noise channel in marginal encoder')
        parser.add_argument('--marginal-encoder-additional-noise-sigma', type=float, default=0.0, metavar='SNE',
                            help='sigma for additional Gaussian noise at marginal encoder')
        #   discriminator randomness
        parser.add_argument('--discriminator-noise-dim', type=int, default=0, metavar='DND',
                            help='dimensionality of additional noise fed to discriminator (default: 0)')
        parser.add_argument('--discriminator-add-noise-channel', action='store_true',
                            help='whether to add noise channel in discriminator')
        parser.add_argument('--discriminator-additional-noise-sigma-xy', type=float, default=0.25, metavar='SNDZ',
                            help='sigma for additional Gaussian noise at discriminators (x, y)')
        parser.add_argument('--discriminator-additional-noise-sigma-z', type=float, default=0.25, metavar='SNDZ',
                            help='sigma for additional Gaussian noise at discriminators (z)')
        parser.add_argument('--discriminator-additional-noise-sigma-uv', type=float, default=0.25, metavar='SNDUV',
                            help='sigma for additional Gaussian noise at discriminators (u, v)')
        parser.add_argument("--discriminator-turn-off-additional-gaussian-noise-during-training", action='store_true', default=False)
        #   the epsilon hack for small discriminator ratios
        parser.add_argument('--additional-epsilon-to-ratio', type=float, default=0)

        # training hyperparameters (general)
        parser.add_argument('--no-cuda', action='store_true', help='disable CUDA use')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for data (default: 64)')
        parser.add_argument('--gener-lr', type=float, default=1e-4, metavar='GenerLR', help='generator learning rate (default: 1e-4)')
        parser.add_argument('--discr-lr', type=float, default=1e-4, metavar='DiscrLR', help='discriminator learning rate (default: 1e-4)')
        parser.add_argument('--n-epochs', type=int, default=100, metavar='E',
                            help='number of epochs to train (default: 100)')
        parser.add_argument('--patience', type=int, default=100)
        parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'rmsprop'])
        parser.add_argument('--beta1', type=float, default=.5)
        parser.add_argument('--beta2', type=float, default=.999)
        parser.add_argument('--clip-grad-norm', type=float, default=-1)
        parser.add_argument('--use-ema', type=str2bool, default=True, help='use exponential moving average in training')
        parser.add_argument('--ema-decay', type=float, default=0.999)
        parser.add_argument('--continue-training-after-nan', action='store_true')

        # adversarial training
        parser.add_argument('--discr-steps', type=int, default=1)
        parser.add_argument('--gener-steps', type=int, default=1)

        # objective hyperparameters (adversarial Wyner model specifics)
        parser.add_argument('--discr-loss-fn', type=str, default='js_var',
                            help='loss function for density ratio matching',
                            choices=['js_var', 'symmetric_kl_var_dv', 'symmetric_kl_var_njw', 'lc_var'])
        parser.add_argument('--gener-loss-fn', type=str, default='symmetric_kl_plugin',
                            help='loss function for distribution matching',
                            choices=['js_plugin', 'symmetric_kl_plugin', 'reverse_kl_plugin', 'forward_kl_plugin'])
        parser.add_argument('--positive-label-smoothing', type=float, default=1,
                            help='positive label smoothing argument in discriminator JSD loss')
        #     discriminator
        parser.add_argument('--uniform-discr-weights', type=str2bool, default=True)
        #     variational regularization
        parser.add_argument('--lambda-var-ci', type=float, default=0,
                            help='loss weight for I(X,Y;Z) regularization (encoder)')
        parser.add_argument('--lambda-var-agg', type=float, default=0,
                            help='loss weight for Z regularization (encoder)')
        #     joint
        parser.add_argument('--lambda-j', type=float, default=0, help='loss weight for joint loss')
        parser.add_argument('--lambda-j-ci', type=float, default=0,
                            help='loss weight for I(X,Y;Z) regularization (joint)')
        #     cond (x->y)
        parser.add_argument('--lambda-cx2y', type=float, default=0, help='loss weight for conditional loss (x->y)')
        parser.add_argument('--lambda-cx2y-ci', type=float, default=0,
                            help='loss weight for I(X,Y;Z) regularization (x->y)')
        parser.add_argument('--lambda-cx2y-agg', type=float, default=0,
                            help='loss weight for Z regularization (x->y)')
        #     cond (y->x)
        parser.add_argument('--lambda-cy2x', type=float, default=0, help='loss weight for conditional loss (y->x)')
        parser.add_argument('--lambda-cy2x-ci', type=float, default=0,
                            help='loss weight for I(X,Y;Z) regularization (y->x)')
        parser.add_argument('--lambda-cy2x-agg', type=float, default=0,
                            help='loss weight for Z regularization (y->x)')
        #     cross (x->y, y->x)
        parser.add_argument('--lambda-c', type=float, default=0, help='loss weight for cross matching loss')
        parser.add_argument('--lambda-c-agg', type=float, default=0, help='loss weight for cross latent matching loss')
        #     marginals (x->x; y->y)
        parser.add_argument('--lambda-mx', type=float, default=0, help='loss weight for marginal x loss')
        parser.add_argument('--lambda-my', type=float, default=0, help='loss weight for marginal y loss')
        #     reconstruction
        parser.add_argument('--lambda-j-rec-x', type=float, default=0,
                            help='loss weight for reconstruction loss (j x)')
        parser.add_argument('--lambda-j-rec-y', type=float, default=0,
                            help='loss weight for reconstruction loss (j y)')
        parser.add_argument('--lambda-j-rec-zuv', type=float, default=0,
                            help='loss weight for reconstruction loss (j z,u,v)')
        parser.add_argument('--lambda-cx2y-rec-y', type=float, default=0,
                            help='loss weight for reconstruction loss (cx2y y)')
        parser.add_argument('--lambda-cy2x-rec-x', type=float, default=0,
                            help='loss weight for reconstruction loss (cy2x x)')
        parser.add_argument('--lambda-mx-rec-x', type=float, default=0,
                            help='loss weight for reconstruction loss ("marginal reconstruction" x)')
        parser.add_argument('--lambda-my-rec-y', type=float, default=0,
                            help='loss weight for reconstruction loss ("marginal reconstruction" y)')

        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
