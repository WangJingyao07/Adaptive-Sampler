from src.datasets.meta_dataset import config
from src.datasets.meta_dataset import pipeline
from src.datasets.meta_dataset import learning_spec
from src.datasets.meta_dataset import dataset_spec as dataset_spec_lib
import tensorflow.compat.v1 as tf
import os
import gin
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet the TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet the TensorFlow warnings


class MetaDatasetReader:
    """
    Class that wraps the Meta-Dataset episode reader.
    """

    def __init__(self, data_path, train_set, validation_set, test_set, max_way_train=5, max_way_test=5,
                 max_support_train=1, max_support_test=15, mode="train_test"):

        self.data_path = data_path
        self.train_dataset_next_task = None
        self.validation_set_dict = {}
        self.test_set_dict = {}

        # Unnecessary Variables
        self.class_augmentations = []
        self.transform = []
        self.target_transform = None
        self.dataset_transform = None
        self.meta_train = False
        self.meta_val = False
        self.meta_test = False
        #

        tf.compat.v1.disable_eager_execution()
        self.session = tf.compat.v1.Session()
        gin.parse_config_file('src/datasets/meta_dataset_config.gin')

        if mode == 'train' or mode == 'train_test':
            train_episode_description = self._get_train_episode_description(
                max_way_train, max_support_train)
            self.train_dataset_next_task = self._init_multi_source_dataset(train_set, learning_spec.Split.TRAIN,
                                                                           train_episode_description)

            test_episode_description = self._get_test_episode_description(
                max_way_test, max_support_test)
            for item in validation_set:
                next_task = self.validation_dataset = self._init_single_source_dataset(item, learning_spec.Split.VALID,
                                                                                       test_episode_description)
                self.validation_set_dict[item] = next_task

        if mode == 'test' or mode == 'train_test':
            test_episode_description = self._get_test_episode_description(
                max_way_test, max_support_test)
            for item in test_set:
                next_task = self._init_single_source_dataset(
                    item, learning_spec.Split.TEST, test_episode_description)
                self.test_set_dict[item] = next_task

    def _init_multi_source_dataset(self, items, split, episode_description):
        dataset_specs = []
        for dataset_name in items:
            dataset_records_path = os.path.join(self.data_path, dataset_name)
            dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
            dataset_specs.append(dataset_spec)

        use_bilevel_ontology_list = [False] * len(items)
        use_dag_ontology_list = [False] * len(items)
        # Enable ontology aware sampling for Omniglot and ImageNet.

        # if 'omniglot' in items:
        #     use_bilevel_ontology_list[items.index('omniglot')] = True
        # if 'ilsvrc_2012' in items:
        #     use_dag_ontology_list[items.index('ilsvrc_2012')] = True

        multi_source_pipeline = pipeline.make_multisource_episode_pipeline(
            dataset_spec_list=dataset_specs,
            use_dag_ontology_list=use_dag_ontology_list,
            use_bilevel_ontology_list=use_bilevel_ontology_list,
            split=split,
            episode_descr_config=episode_description,
            image_size=84,
            shuffle_buffer_size=1000)

        iterator = multi_source_pipeline.make_one_shot_iterator()
        return iterator.get_next()

    def _init_single_source_dataset(self, dataset_name, split, episode_description):
        dataset_records_path = os.path.join(self.data_path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)

        # Enable ontology aware sampling for Omniglot and ImageNet.
        use_bilevel_ontology = False
        # if 'omniglot' in dataset_name:
        #     use_bilevel_ontology = True

        use_dag_ontology = False
        # if 'ilsvrc_2012' in dataset_name:
        #     use_dag_ontology = True

        single_source_pipeline = pipeline.make_one_source_episode_pipeline(
            dataset_spec=dataset_spec,
            use_dag_ontology=use_dag_ontology,
            use_bilevel_ontology=use_bilevel_ontology,
            split=split,
            episode_descr_config=episode_description,
            image_size=84,
            shuffle_buffer_size=1000)

        iterator = single_source_pipeline.make_one_shot_iterator()
        return iterator.get_next()

    def _get_task(self, next_task):
        (episode, source_id) = self.session.run(next_task)
        task_dict = {
            'context_images': episode[0],
            'context_labels': episode[1],
            'context_tasks': episode[2],
            'target_images': episode[3],
            'target_labels': episode[4],
            'target_tasks': episode[5]
        }
        return task_dict

    def get_train_task(self):
        return self._get_task(self.train_dataset_next_task)

    def get_validation_task(self, item):
        return self._get_task(self.validation_set_dict[item])

    def get_test_task(self, item):
        return self._get_task(self.test_set_dict[item])

    def _get_train_episode_description(self, max_way_train, max_support_train):
        return config.EpisodeDescriptionConfig(
            num_ways=5,
            num_support=1,
            num_query=15,
            min_ways=5,
            max_ways_upper_bound=max_way_train,
            max_num_query=15,
            max_support_set_size=max_support_train,
            max_support_size_contrib_per_class=100,
            min_log_weight=-0.69314718055994529,  # np.cnaps_layer_log.txt(0.5)
            max_log_weight=0.69314718055994529,  # np.cnaps_layer_log.txt(2)
            ignore_dag_ontology=False,
            ignore_bilevel_ontology=False,
            ignore_hierarchy_probability=0.0,
            simclr_episode_fraction=0.0,
            min_examples_in_class=16
        )

    def _get_test_episode_description(self, max_way_test, max_support_test):
        return config.EpisodeDescriptionConfig(
            num_ways=5,
            num_support=1,
            num_query=15,
            min_ways=5,
            max_ways_upper_bound=max_way_test,
            max_num_query=15,
            max_support_set_size=max_support_test,
            max_support_size_contrib_per_class=100,
            min_log_weight=-0.69314718055994529,  # np.cnaps_layer_log.txt(0.5)
            max_log_weight=0.69314718055994529,  # np.cnaps_layer_log.txt(2)
            ignore_dag_ontology=False,
            ignore_bilevel_ontology=False,
            ignore_hierarchy_probability=0.0,
            simclr_episode_fraction=0.0,
            min_examples_in_class=16
        )


class SingleDatasetReader:
    """
    Class that wraps the Meta-Dataset episode reader to read in a single dataset.
    """

    def __init__(self, data_path, mode, dataset, way, shot, query_train, query_test):

        self.data_path = data_path
        self.train_next_task = None
        self.validation_next_task = None
        self.test_next_task = None
        tf.compat.v1.disable_eager_execution()
        self.session = tf.compat.v1.Session()
        gin.parse_config_file('src/datasets/meta_dataset_config.gin')

        fixed_way_shot_train = self._get_train_episode_description(
            num_ways=way, num_support=shot, num_query=query_train)
        fixed_way_shot_test = self._get_test_episode_description(
            num_ways=way, num_support=shot, num_query=query_test)

        if mode == 'train' or mode == 'train_test':
            self.train_next_task = self._init_dataset(
                dataset, learning_spec.Split.TRAIN, fixed_way_shot_train)
            self.validation_next_task = self._init_dataset(
                dataset, learning_spec.Split.VALID, fixed_way_shot_test)

        if mode == 'test' or mode == 'train_test':
            self.test_next_task = self._init_dataset(
                dataset, learning_spec.Split.TEST, fixed_way_shot_test)

    def _init_dataset(self, dataset, split, episode_description):
        dataset_records_path = os.path.join(self.data_path, dataset)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)

        single_source_pipeline = pipeline.make_one_source_episode_pipeline(
            dataset_spec=dataset_spec,
            use_dag_ontology=False,
            use_bilevel_ontology=False,
            split=split,
            episode_descr_config=episode_description,
            image_size=84,
            shuffle_buffer_size=1000)

        iterator = single_source_pipeline.make_one_shot_iterator()
        return iterator.get_next()

    def _get_task(self, next_task):
        (episode, source_id) = self.session.run(next_task)
        task_dict = {
            'context_images': episode[0],
            'context_labels': episode[1],
            'context_tasks': episode[2],
            'target_images': episode[3],
            'target_labels': episode[4],
            'target_tasks': episode[5]
        }
        return task_dict

    def get_train_task(self):
        return self._get_task(self.train_next_task)

    def get_validation_task(self, item):
        return self._get_task(self.validation_next_task)

    def get_test_task(self, item):
        return self._get_task(self.test_next_task)

    def _get_train_episode_description(self, num_ways, num_support, num_query):
        return config.EpisodeDescriptionConfig(
            num_ways=num_ways,
            num_support=num_support,
            num_query=num_query,
            min_ways=5,
            max_ways_upper_bound=num_ways,
            max_num_query=num_query,
            max_support_set_size=num_support,
            max_support_size_contrib_per_class=100,
            min_log_weight=-0.69314718055994529,  # np.cnaps_layer_log.txt(0.5)
            max_log_weight=0.69314718055994529,  # np.cnaps_layer_log.txt(2)
            ignore_dag_ontology=False,
            ignore_bilevel_ontology=False,
            ignore_hierarchy_probability=0.0,
            simclr_episode_fraction=0.0,
            min_examples_in_class=16
        )

    def _get_test_episode_description(self, num_ways, num_support, num_query):
        return config.EpisodeDescriptionConfig(
            num_ways=num_ways,
            num_support=num_support,
            num_query=num_query,
            min_ways=5,
            max_ways_upper_bound=num_ways,
            max_num_query=num_query,
            max_support_set_size=num_support,
            max_support_size_contrib_per_class=100,
            min_log_weight=-0.69314718055994529,  # np.cnaps_layer_log.txt(0.5)
            max_log_weight=0.69314718055994529,  # np.cnaps_layer_log.txt(2)
            ignore_dag_ontology=False,
            ignore_bilevel_ontology=False,
            ignore_hierarchy_probability=0.0,
            simclr_episode_fraction=0.0,
            min_examples_in_class=16
        )
