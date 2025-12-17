import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Complete Robotics Education',
    description: (
      <>
        Learn the complete stack of modern robotics: from ROS 2 fundamentals to AI integration,
        simulation environments, and vision-language-action systems.
      </>
    ),
  },
  {
    title: 'Hands-On Learning',
    description: (
      <>
        Build real robotic systems through practical projects that integrate all
        components of the robotics stack.
      </>
    ),
  },
  {
    title: 'Industry-Ready Skills',
    description: (
      <>
        Master the tools and technologies used in cutting-edge robotics research
        and development.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}