import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';

export default function Home() {
  return (
    <Layout
      title="Physical AI & Humanoid Robotics"
      description="A Textbook for Humanoid Robotics"
    >
      <header className={styles.heroBanner}>
        <div className={styles.container}>
          {/* Left side: text content */}
          <div className={styles.left}>
            <h1 className={styles.title}>Physical AI & Humanoid Robotics</h1>
            <p className={styles.subtitle}>
              A complete open-source textbook for the future of robotics.
            </p>
            <div className={styles.buttons}>
              <Link className="button button--primary button--lg" to="/docs/module1-ros2/ch01-intro-to-ros2">
                Start Learning
              </Link>
               
              {/* <Link
                className="button button--secondary button--lg"
                style={{ marginLeft: '1rem' }}
                to="https://github.com/facebook/docusaurus"
              >
                GitHub
              </Link> */}
            </div>
          </div>

          {/* Right side: book image */}
          <div className={styles.right}>
            <img
              src="/img/book-cover.png"
              alt="Physical AI & Humanoid Robotics Book Cover"
              className={styles.bookCover}
            />
          </div>
        </div>
      </header>
    </Layout>
  );
}
