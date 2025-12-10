// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// You can also use `@provide` to get types from other files as well.

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A Textbook',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-docusaurus-site.example.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'facebook', // Usually your GitHub org/user name.
  projectName: 'docusaurus', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  scripts: [
    {
      src: '/rag-chatbot-embed.js',
      async: true,
      defer: true,
    },
    {
      src: '/js/auth-navbar.js',
      async: true,
      defer: true,
    },
  ],

  themes: [
  ],

  plugins: [
    // Add a plugin to configure webpack devServer
    async function configureDevServer(context, options) {
      return {
        name: 'dev-server-config-plugin',
        configureWebpack(config, isServer) {
          if (!isServer) {
            // Only apply this in development mode for client-side builds
            return {
              devServer: {
                proxy: [
                  {
                    context: ['/api', '/auth'],
                    target: process.env.BACKEND_URL || 'http://localhost:8000',
                    changeOrigin: true,
                    secure: false,
                  }
                ]
              }
            };
          }
          return {};
        }
      };
    }
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical Ai & Humanoid Robotics ',
        logo: {
          alt: 'My Site Logo',
          src: 'img/book-logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'TextBook',
          },
        ],
      },
      footer: {
        style: 'light',
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Built with Docusaurus.`,
      },
      prism: {
        theme: require('prism-react-renderer').themes.github,
        darkTheme: require('prism-react-renderer').themes.dracula,
      },
    }),

  // Move the devServer config to customFields
  customFields: {
    devServer: {
      proxy: {
        '/api': {
          target: process.env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true,
          secure: false, // Set to true in production with proper SSL
        },
        '/auth': {
          target: process.env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true,
          secure: false, // Set to true in production with proper SSL
        }
      },
    }
  }
};

module.exports = config;


