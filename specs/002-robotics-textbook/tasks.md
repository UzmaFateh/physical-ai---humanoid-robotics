---
description: "Task list for the Physical AI & Humanoid Robotics Textbook feature"
---

# Tasks: Physical AI & Humanoid Robotics Textbook

**Input**: Design documents from `/specs/002-robotics-textbook/`
**Prerequisites**: plan.md, spec.md

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)

## Path Conventions

- All content will be in the `docs/` directory.
- All code examples will be in the `src/code-examples/` directory.

---

## Phase 1: Setup (Docusaurus Project Initialization)

**Purpose**: Create the basic Docusaurus project and folder structure.

- [x] T001 Initialize a new Docusaurus classic project in the repository root.
- [x] T002 [P] Create the module directory structure: `docs/module1-ros2`, `docs/module2-digital-twin`, `docs/module3-nvidia-isaac`, `docs/module4-vla-systems`.
- [x] T003 [P] Create the code example directory structure: `src/code-examples/module1`, `src/code-examples/module2`, `src/code-examples/module3`, `src/code-examples/module4`.
- [x] T004 Configure `docusaurus.config.js` with the textbook title ("Physical AI & Humanoid Robotics"), theme, and navigation.
- [x] T005 Configure `sidebars.js` to define the chapter order for all four modules.

---

## Phase 2: Foundational (Chapter Placeholders)

**Purpose**: Create all the chapter files to scaffold the entire textbook.

**⚠️ CRITICAL**: No content writing can begin until this phase is complete.

- [x] T006 [P] Create placeholder `ch01-intro-to-ros2.md` in `docs/module1-ros2/`.
- [x] T007 [P] Create placeholder `ch02-concepts.md` in `docs/module1-ros2/`.
- [x] T008 [P] Create placeholder `ch03-packages.md` in `docs/module1-ros2/`.
- [x] T009 [P] Create placeholder `ch04-urdf-launch-files.md` in `docs/module1-ros2/`.
- [x] T010 [P] Create placeholder `ch05-robot-description.md` in `docs/module2-digital-twin/`.
- [x] T011 [P] Create placeholder `ch06-gazebo-physics.md` in `docs/module2-digital-twin/`.
- [x] T012 [P] Create placeholder `ch07-unity-viz.md` in `docs/module2-digital-twin/`.
- [x] T013 [P] Create placeholder `ch08-integration.md` in `docs/module2-digital-twin/`.
- [x] T014 [P] Create placeholder `ch09-isaac-sim-fundamentals.md` in `docs/module3-nvidia-isaac/`.
- [x] T015 [P] Create placeholder `ch10-isaac-ros.md` in `docs/module3-nvidia-isaac/`.
- [x] T016 [P] Create placeholder `ch11-navigation.md` in `docs/module3-nvidia-isaac/`.
- [x] T017 [P] Create placeholder `ch12-rl-sim-to-real.md` in `docs/module3-nvidia-isaac/`.
- [x] T018 [P] Create placeholder `ch13-whisper.md` in `docs/module4-vla-systems/`.
- [x] T019 [P] Create placeholder `ch14-llm-planning.md` in `docs/module4-vla-systems/`.
- [x] T020 [P] Create placeholder `ch15-plans-to-actions.md` in `docs/module4-vla-systems/`.
- [x] T021 [P] Create placeholder `ch16-capstone.md` in `docs/module4-vla-systems/`.

**Checkpoint**: Foundation ready - content implementation can now begin.

---

## Phase 3: User Story 1 - ROS 2 Fundamentals (Priority: P1) 🎯 MVP

**Goal**: Deliver the first module of the textbook, teaching students the fundamentals of ROS 2.
**Independent Test**: A student can read Module 1 and complete the labs to build and run basic `rclpy` applications.

### Implementation for User Story 1

- [x] T022 [US1] Write Chapter 1 content (Introduction to ROS 2) in `docs/module1-ros2/ch01-intro-to-ros2.md`.
- [x] T023 [US1] Write Chapter 2 content (Nodes, Topics, Services, Actions) in `docs/module1-ros2/ch02-concepts.md`.
- [x] T024 [P] [US1] Add `rclpy` code examples for a publisher and subscriber to `src/code-examples/module1/`.
- [x] T025 [US1] Write Chapter 3 content (ROS 2 Packages with Python) in `docs/module1-ros2/ch03-packages.md`.
- [x] T026 [US1] Write Chapter 4 content (URDF & Launch Files) in `docs/module1-ros2/ch04-urdf-launch-files.md`.
- [x] T027 [P] [US1] Create a sample URDF and launch file for a simple robot in `src/code-examples/module1/`.
- [x] T028 [US1] Write lab assignments for all chapters in Module 1.

**Checkpoint**: Module 1 is complete and ready for student use.

---

## Phase 4: User Story 2 - Simulating Robotic Environments (Priority: P1)

**Goal**: Deliver the second module, teaching students to simulate robots in Gazebo and Unity.
**Independent Test**: A student can read Module 2 and complete the labs to simulate a robot and integrate it with ROS 2.

### Implementation for User Story 2

- [x] T029 [US2] Write Chapter 5 content (Robot Description) in `docs/module2-digital-twin/ch05-robot-description.md`.
- [x] T030 [US2] Write Chapter 6 content (Gazebo Physics & Sensors) in `docs/module2-digital-twin/ch06-gazebo-physics.md`.
- [x] T031 [P] [US2] Add code examples for spawning a robot in Gazebo to `src/code-examples/module2/`.
- [x] T032 [US2] Write Chapter 7 content (Unity Visualization) in `docs/module2-digital-twin/ch07-unity-viz.md`.
- [x] T033 [US2] Write Chapter 8 content (Integrating Gazebo/Unity with ROS 2) in `docs/module2-digital-twin/ch08-integration.md`.
- [x] T034 [P] [US2] Add code examples for the ROS-Unity TCP connector to `src/code-examples/module2/`.
- [x] T035 [US2] Write lab assignments for all chapters in Module 2.

**Checkpoint**: Module 2 is complete and ready for student use.

---

## Phase 5: User Story 3 - Developing AI Robot Control (Priority: P2)

**Goal**: Deliver the third module, teaching students to use NVIDIA Isaac for perception and navigation.
**Independent Test**: A student can read Module 3 and complete the labs to run basic AI simulations in Isaac Sim.

### Implementation for User Story 3

- [x] T036 [US3] Write Chapter 9 content (Isaac Sim Fundamentals) in `docs/module3-nvidia-isaac/ch09-isaac-sim-fundamentals.md`.
- [x] T037 [US3] Write Chapter 10 content (Isaac ROS) in `docs/module3-nvidia-isaac/ch10-isaac-ros.md`.
- [x] T038 [P] [US3] Add code examples for Isaac ROS perception to `src/code-examples/module3/`.
- [x] T039 [US3] Write Chapter 11 content (Navigation with Nav2) in `docs/module3-nvidia-isaac/ch11-navigation.md`.
- [x] T040 [US3] Write Chapter 12 content (RL + Sim-to-Real Transfer) in `docs/module3-nvidia-isaac/ch12-rl-sim-to-real.md`.
- [x] T041 [US3] Write lab assignments for all chapters in Module 3.

**Checkpoint**: Module 3 is complete and ready for student use.

---

## Phase 6: User Story 4 - Implementing Vision-Language-Action (Priority: P2)

**Goal**: Deliver the final module, teaching students to build VLA systems with Whisper and LLMs.
**Independent Test**: A student can read Module 4 and complete the labs to control a simulated robot with voice commands.

### Implementation for User Story 4

- [x] T042 [US4] Write Chapter 13 content (Whisper Voice Command Integration) in `docs/module4-vla-systems/ch13-whisper.md`.
- [x] T043 [P] [US4] Add code examples for using the Whisper API to `src/code-examples/module4/`.
- [x] T044 [US4] Write Chapter 14 content (LLM Cognitive Planning) in `docs/module4-vla-systems/ch14-llm-planning.md`.
- [x] T045 [US4] Write Chapter 15 content (Converting Plans to ROS 2 Actions) in `docs/module4-vla-systems/ch15-plans-to-actions.md`.
- [x] T046 [P] [US4] Add code examples for using an LLM API to generate a task plan to `src/code-examples/module4/`.
- [x] T047 [US4] Write Chapter 16 content (Capstone: Autonomous Humanoid Pipeline) in `docs/module4-vla-systems/ch16-capstone.md`.
- [x] T048 [US4] Write lab assignments for all chapters in Module 4.

**Checkpoint**: Module 4 is complete and ready for student use.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Finalize the textbook for release.

- [ ] T049 [P] Add a section with IEEE-style references to all relevant chapters.
- [ ] T050 Review all 16 chapters for technical accuracy, clarity, and consistency.
- [ ] T051 [P] Lint all Python code in `src/code-examples/` using `black` and `ruff`.
- [ ] T052 Validate the full Docusaurus build by running `yarn build` and ensure there are no errors.
- [ ] T053 Create a final distributable archive of the textbook project.

---

## Dependencies & Execution Order

- **Phase 1 (Setup)** must be completed before all other phases.
- **Phase 2 (Foundational)** must be completed before all content-writing phases (3-6).
- **User Story Phases (3-6)** can be completed in parallel after Phase 2 is done.
- **Phase 7 (Polish)** can only begin after all desired User Story phases are complete.

## Implementation Strategy

### MVP First (Module 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test that Module 1 is complete, and the Docusaurus site builds and serves the content correctly.
