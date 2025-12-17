#!/bin/bash
# deployment script for Docusaurus site

set -e # Exit with nonzero exit code if anything fails

SOURCE_BRANCH="main"
TARGET_BRANCH="gh-pages"

function doCompile {
  npm run build
}

# Pull requests and commits to other branches shouldn't try to deploy
if [ "$TRAVIS_PULL_REQUEST" != "false" -o "$TRAVIS_BRANCH" != "$SOURCE_BRANCH" ]; then
    echo "Skipping deploy; just doing a build."
    doCompile
    exit 0
fi

# Save current commit hash
CURRENT_COMMIT=$(git rev-parse HEAD)

# Clone the existing gh-pages for this repo into out/
# Create a new empty branch from the cloned content
git clone -b $TARGET_BRANCH https://github.com/$TRAVIS_REPO_SLUG.git out
cd out/
git checkout $TARGET_BRANCH

# Now let's go have some fun with the cloned repo
git config user.name "Travis CI"
git config user.email "$COMMIT_AUTHOR_EMAIL"

# Clean out existing contents
git rm -rf .

# Copy the new build
cd ..
doCompile
cp -a build/. ../out/

# Now let's go back to the cloned repo and commit/push
cd out/

git add -A .
git commit --allow-empty -m "Deploy to GitHub Pages: ${CURRENT_COMMIT}"

# Get the deploy key by using Travis's stored variables to decrypt deploy_key.enc
ENCRYPTED_KEY_VAR="encrypted_${ENCRYPTION_LABEL}_key"
ENCRYPTED_IV_VAR="encrypted_${ENCRYPTION_LABEL}_iv"
ENCRYPTED_KEY=${!ENCRYPTED_KEY_VAR}
ENCRYPTED_IV=${!ENCRYPTED_IV_VAR}
openssl aes-256-cbc -K $ENCRYPTED_KEY -iv $ENCRYPTED_IV -in ../deploy_key.enc -out ../deploy_key -d
chmod 600 ../deploy_key
eval $(ssh-agent -s)
ssh-add ../deploy_key

# Now that we're all set up, we can push.
git push origin $TARGET_BRANCH