from tensor2tensor.utils import registry
from tensor2tensor.data_generators.wmt import token_generator
from tensor2tensor.data_generators import generator_utils, problem
from tensor2tensor.data_generators import text_encoder

EOS = text_encoder.EOS_ID

@registry.register_problem
class AutoencodeTweets(problem.Text2TextProblem):

  @property
  def targeted_vocab_size(self):
    return 2**14 #16384

  @property
  def is_character_level(self):
    return False

  @property
  def num_shards(self):
    return 3

  @property
  def vocab_name(self):
    return "vocab.txt"

  @property
  def use_subword_tokenizer(self):
    return True

  def generator(self, data_dir, tmp_dir, train):
    root = '/Users/jmugan/Dropbox/presentations/2017_TensorFlow_for_NLP/code/data/'
    source = root+'tweets.txt'
    target = root+'tweets.txt'
    filepatterns = [source] # to make vocab
    symbolizer_vocab = generator_utils.get_or_generate_txt_vocab(data_dir, self.vocab_file,
                                      self.targeted_vocab_size,filepatterns=filepatterns)
    return token_generator(source, target,
                           symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
      return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
      return problem.SpaceID.EN_TOK

