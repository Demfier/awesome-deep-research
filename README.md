# 🧠 [Awesome Deep Research](https://github.com/Demfier/awesome-deep-research) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub stars](https://img.shields.io/github/stars/Demfier/awesome-deep-research?style=social) ![GitHub forks](https://img.shields.io/github/forks/Demfier/awesome-deep-research?style=social) ![License](https://img.shields.io/badge/license-MIT-blue)

<div align="center">
  <img src="./hero_figure.png" width="400" alt="Deep Research Hero Image">
  <br><br>
  <p><b>A curated list of advanced AI research agents, benchmarks, and evaluation frameworks</b></p>
  <p>Contribute and explore cutting-edge tools for AI research</p>
</div>

## 📋 Overview

Welcome to **Awesome Deep Research**! This curated list focuses on tools, agents, benchmarks, and resources specifically designed for **deep research** - a cutting-edge capability of AI systems to autonomously perform comprehensive research tasks that previously required human experts.

## 📋 What is Deep Research?

**Deep Research** refers to AI systems that can autonomously perform multi-step research tasks, much like a human expert gathering information from diverse sources and synthesizing insights. These agents leverage powerful language models with reasoning abilities to:

- Search through hundreds of online sources
- Analyze large amounts of information
- Synthesize findings into comprehensive reports with citations
- Execute complex research workflows autonomously
- Iteratively refine search queries based on initial findings
- Generate well-structured, nuanced reports with proper attribution

A Deep Research agent takes a prompt and conducts thorough investigation across multiple sources, just like an expert research analyst would. The result is typically a detailed report that feels like it was written by a domain expert.

---

## 📚 Contents

- [🧠 Awesome Deep Research](#-awesome-deep-research)
  - [📋 Overview](#-overview)
  - [📋 What is Deep Research?](#-what-is-deep-research)
  - [📚 Contents](#-contents)
  - [🤖 Deep Research Systems](#-deep-research-systems)
    - [🔓 Commercial Deep Research Platforms](#-commercial-deep-research-platforms)
    - [🌐 Open-Source Deep Research Implementations](#-open-source-deep-research-implementations)
    - [🔬 Specialized Research Assistants](#-specialized-research-assistants)
  - [🤖 Agent Technologies for Research](#-agent-technologies-for-research)
    - [🤖 General-Purpose AI Agents](#-general-purpose-ai-agents)
  - [📊 Benchmarks & Evaluation](#-benchmarks--evaluation)
  - [📚 Tutorials & Guides](#-tutorials--guides)
  - [📊 Datasets for Deep Research](#-datasets-for-deep-research)
  - [📝 Related Papers and Resources](#-related-papers-and-resources)
    - [📄 Deep Research Foundations](#-deep-research-foundations)
    - [📄 Evaluation & Benchmarking](#-evaluation--benchmarking)
    - [📄 Literature Review Systems](#-literature-review-systems)
    - [📄 Agent Systems & Collaboration](#-agent-systems--collaboration)
  - [🤝 Contributions](#-contributions)
  - [📜 License](#-license)

---

## 🤖 Deep Research Systems

<div align="center">
  <p><i>AI systems specialized in autonomous research, synthesis, and report generation</i></p>
</div>

### 🔓 Commercial Deep Research Platforms

- **[OpenAI's Deep Research](https://openai.com/index/introducing-deep-research/)** - An AI agent that uses reasoning to synthesize large amounts of online information and complete multi-step research tasks.

- **[Gemini's Deep Research](https://gemini.google/overview/deep-research/)** - An agentic feature in Gemini that automatically browses websites, thinks through findings, and creates insightful multi-page reports.

- **[Perplexity's Deep Research](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)** - Allows users to conduct in-depth research and analysis, delivering comprehensive reports.

- **[Manus AI](https://manus.ai/)** - An autonomous AI agent capable of independently carrying out complex real-world tasks.

- **[Agents Inc](https://www.agents.inc/)** - Provides specialized AI agents for tasks such as Global News Radar, Scientific Knowledge, and Patent Analysis, particularly useful for research in science and policy domains.

- **[Pandi](https://askpandi.com/ask)** - An answer engine that compiles search results into concise webpages, providing multi-modal answers with citations, ideal for quick research synthesis.

### 🌐 Open-Source Deep Research Implementations

- **[OpenManus](https://github.com/mannaandpoem/OpenManus)** - An open-source alternative to Manus AI, allowing users to create AI agents without invite codes.

- **[Autogen](https://github.com/microsoft/autogen)** - A framework for developing multi-agent conversation systems.

- **[Crew AI](https://github.com/crewAIInc/crewAI)** - A framework for building collaborative AI agents.

- **[OpenHands (fka OpenDevin)](https://github.com/All-Hands-AI/OpenHands)** - An agentic AI software engineer that can understand high-level instructions and write code.

- **[LocalGPT](https://github.com/PromtEngineer/localGPT)** - Allows conversing with documents locally without compromising privacy.

- **[DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)** - A framework for scaling deep research via reinforcement learning in real-world environments.

- **[Search-R1](https://github.com/PeterGriffinJin/Search-R1)** - An efficient RL training framework for reasoning & search engine calling for LLMs.

- **[OpenManus-RL](https://github.com/OpenManus/OpenManus-RL)** - Explores reinforcement learning-based tuning methods for LLM agents.

- **[XAgent](https://github.com/OpenBMB/XAgent)** - An open-source experimental LLM-driven autonomous agent for solving various tasks, featuring autonomy, safety, and a dual-loop mechanism.

- **[WrenAI](https://github.com/Canner/WrenAI)** - An open-source AI-powered data assistant for retrieving results and insights without SQL, useful for non-programmers in research.

- **[Local Deep Researcher](https://github.com/langchain-ai/local-deep-researcher)** - A fully local web research assistant that uses LLMs hosted by Ollama or LMStudio to generate search queries, gather results, summarize findings, and iterate through multiple research cycles.

- **[DeepSearcher](https://github.com/zilliztech/deep-searcher)** - Combines cutting-edge LLMs and Vector Databases to perform search, evaluation, and reasoning based on private data, providing highly accurate answers for enterprise knowledge management.

- **[Deep Research](https://github.com/dzhng/deep-research)** - A minimal implementation (<500 LOC) of a deep research agent that combines search engines, web scraping, and LLMs to perform iterative research on any topic.

- **[OpenDeepResearcher](https://github.com/mshumer/OpenDeepResearcher)** - An AI researcher that continuously searches for information based on a user query, using SERPAPI for Google searches, Jina for webpage content extraction, and OpenRouter (default: claude-3.5-haiku) for generating search queries and evaluating relevance.

- **[Deep Research Agent with Motia Framework](https://github.com/MotiaDev/motia-examples/tree/main/examples/ai-deep-research-agent)** - A powerful research assistant that leverages the Motia Framework to perform comprehensive web research on any topic and question.

### 🔬 Specialized Research Assistants

- **[Elicit](https://elicit.org/)** - AI-powered literature review and research automation.

- **[PaperQA](https://paperqa.ai/)** - AI-driven paper-based Q&A system for research.

- **[ResearchRabbit](https://researchrabbitapp.com/)** - AI tool for discovering and tracking academic papers.

- **[CAMEL](https://camel-ai.org/)** - Cooperative AI agents working on research-driven tasks.

- **[MetaGPT](https://github.com/geekan/MetaGPT)** - Multi-agent framework for collaborative AI research.

- **[LitLLM](https://litllm.github.io/)** - A toolkit for leveraging large language models to assist in scientific literature reviews, helping researchers identify, synthesize, and contextualize relevant prior work more efficiently.

## 🤖 Agent Technologies for Research

<div align="center">
  <p><i>Supporting technologies that enable or enhance deep research capabilities</i></p>
</div>

### 🤖 General-Purpose AI Agents

General-Purpose AI Agents are autonomous systems designed to execute a broad spectrum of tasks across various domains, leveraging language models for reasoning and decision-making. These agents can handle task planning, execution, and interaction with diverse tools and APIs.

- **[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)** - An open-source autonomous AI agent employing GPT-4 to tackle complex tasks by breaking them into subtasks.

- **[BabyAGI](https://github.com/yoheinakajima/babyagi)** - A task management system that creates and executes tasks based on user-defined objectives.

- **[AgentGPT](https://github.com/reworkd/AgentGPT)** - An open-source platform for creating and deploying autonomous AI agents with pre-built templates.

- **[Superagent](https://github.com/homanp/superagent)** - A tool for building personalized AI assistants specialized in web research.

- **[OpenAgents](https://github.com/xlang-ai/OpenAgents)** - A platform for creating, hosting, and managing AI agents with user-friendly interfaces.

- **[CAMEL AI](https://github.com/camel-ai/camel)** - An open-source research community exploring agent scaling laws for collaboration.

### 🛠️ Agent Frameworks

<div align="center">
  <p><i>Foundational tools for building and deploying research-capable agents</i></p>
</div>

- **[LangChain](https://github.com/langchain-ai/langchain)** - A framework for building applications powered by large language models, with components for research workflows.

- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Framework for building stateful, multi-actor LLM applications extending LangChain.

- **[veRL](https://github.com/volcengine/verl)** - A flexible and efficient reinforcement learning library for large language models, used by several deep research projects.

---

## 📊 Benchmarks & Evaluation

<div align="center">
  <p><i>Standards and frameworks for measuring deep research capabilities and performance</i></p>
</div>

<div align="center">
  <table>
    <tr>
      <th>Name</th>
      <th>Description</th>
      <th>Link</th>
    </tr>
    <tr>
      <td><b>GAIA</b></td>
      <td>A benchmark for general AI assistants with real-world questions requiring reasoning, multi-modality, web browsing and tool-use</td>
      <td><a href="https://huggingface.co/spaces/gaia-benchmark/leaderboard">Leaderboard</a></td>
    </tr>
    <tr>
      <td><b>HELM</b></td>
      <td>Holistic Evaluation of Language Models across various dimensions</td>
      <td><a href="https://crfm.stanford.edu/helm/">Link</a></td>
    </tr>
    <tr>
      <td><b>BIG-bench</b></td>
      <td>A collaborative benchmark with challenging tasks for language models</td>
      <td><a href="https://github.com/google/BIG-bench">GitHub</a></td>
    </tr>
    <tr>
      <td><b>AGIEval</b></td>
      <td>A human-centric benchmark for evaluating general AI capabilities</td>
      <td><a href="https://arxiv.org/abs/2304.06364">Paper</a></td>
    </tr>
    <tr>
      <td><b>BrowseComp</b></td>
      <td>A benchmark by OpenAI specifically designed to evaluate AI agents' ability to browse the web and find hard-to-find, interconnected information. Emphasizes persistence and creativity in information seeking.</td>
      <td><a href="https://openai.com/index/browsecomp/">Link</a></td>
    </tr>
    <tr>
      <td><b>τ-Bench</b></td>
      <td>Evaluates AI agents' performance in real-world settings with dynamic user and tool interaction</td>
      <td><a href="https://sierra.ai/blog/benchmarking-ai-agents">Link</a></td>
    </tr>
    <tr>
      <td><b>ITBench</b></td>
      <td>For evaluating IT automation agents</td>
      <td><a href="https://github.com/IBM/itbench">GitHub</a></td>
    </tr>
    <tr>
      <td><b>MLAgentBench</b></td>
      <td>For evaluating language agents on machine learning experimentation</td>
      <td><a href="https://arxiv.org/abs/2310.03302">arXiv</a></td>
    </tr>
    <tr>
      <td><b>WebArena</b></td>
      <td>A realistic web environment for building autonomous agents</td>
      <td><a href="https://webarena.dev/">Link</a></td>
    </tr>
    <tr>
      <td><b>OSWorld</b></td>
      <td>A scalable, real computer environment for multimodal agents</td>
      <td><a href="https://github.com/xlang-ai/OSWorld">GitHub</a></td>
    </tr>
    <tr>
      <td><b>AgentBench</b></td>
      <td>Evaluates multi-skill capabilities of LLMs as agents in eight different environments</td>
      <td><a href="https://llmbench.ai/">Link</a></td>
    </tr>
    <tr>
      <td><b>OpenAI Evals</b></td>
      <td>Framework for evaluating language models with customizable metrics and fine-tuning support</td>
      <td><a href="https://github.com/openai/evals">GitHub</a></td>
    </tr>
  </table>
</div>

---

## 📚 Tutorials & Guides

<div align="center">
  <p><i>Learning resources for building and using deep research systems</i></p>
</div>

- **[AutoGPT Tutorial: Building an AI Research Assistant](https://lablab.ai/t/autogpt-tutorial-creating-a-research-assistant-with-auto-gpt-forge)** - A step-by-step guide using AutoGPT to create a research assistant that generates reports.

- **[Building a Group of AI Researchers in 21 mins](https://www.ai-jason.com/learning-ai/how-to-build-ai-agent-tutorial-3)** - Guide for creating multiple GPT assistants for research using Autogen.

- **[The Complete Guide to Building Your First AI Agent with LangGraph](hhttps://medium.com/data-science-collective/the-complete-guide-to-building-your-first-ai-agent-its-easier-than-you-think-c87f376c84b2)** - A detailed guide to creating a text analysis agent for research.

- **[Together's Open Deep Research Cookbook](https://github.com/togethercomputer/together-cookbook/blob/main/Agents/Together_Open_Deep_Research_CookBook.ipynb)** - An efficient open-source implementation tutorial for deep research with multi-step web search.

- **[DeepResearch: Building a Research Automation App with Dify](https://dify.ai/blog/deepresearch-building-a-research-automation-app-with-dify)** - A guide explaining how to automate multi-step searches and summarization using Dify agentic workflow, with iteration nodes that loop through search rounds.

- **[Building Agentic RAG with LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/)** - A DeepLearningAI course on building research assistants that can reason over multiple documents and answer complex questions, covering routing, tool use, multi-step reasoning with memory, and debugging.

- **[Browser-Use Web UI for Deep Research](https://github.com/browser-use/web-ui)** - An open-source browser interface powered by Gradio that enables deep research functionality.

---

## 📊 Datasets for Deep Research

<div align="center">
  <p><i>Essential datasets for training and evaluating research capabilities in AI agents</i></p>
</div>

- **[HotpotQA](https://hotpotqa.github.io/)** - A dataset for multi-hop question answering across documents, requiring reasoning over multiple sources.

- **[Qasper](https://allenai.org/data/qasper)** - A dataset for question answering over scientific papers, ideal for evaluating research agents in academic contexts.

- **[Natural Questions](https://ai.google.com/research/NaturalQuestions/)** - A benchmark for open-domain question answering, testing agents' ability to find information from the web.

- **[MS MARCO](https://microsoft.github.io/msmarco/)** - Machine reading comprehension dataset, useful for evaluating information retrieval capabilities.

- **[FEVER](https://fever.ai/)** - Fact Extraction and VERification dataset, for evaluating fact-checking in research.

---

## 📝 Related Papers and Resources

<div align="center">
  <p><i>Academic works and documentation advancing the field of deep research</i></p>
</div>

### 📄 Deep Research Foundations

- 📄 [OpenAI Deep Research System Card](https://cdn.openai.com/deep-research-system-card.pdf) - Technical details of OpenAI's deep research system architecture and capabilities.

- 📄 [Gemini Deep Research Overview](https://gemini.google/overview/deep-research/?hl=en-CA) - Detailed explanation of Google's approach to deep research with Gemini.

- 📄 [Open Deep Search: Democratizing Search with Open-source Reasoning Agents](https://arxiv.org/abs/2503.20201) - Introduces Open Deep Search (ODS), using reasoning agents with web search tools.

- 📄 [Agentic Reasoning: Reasoning LLMs with Tools for Deep Research](https://arxiv.org/abs/2502.04644) - Presents a framework enhancing LLMs with tools for deep research, outperforming on PhD-level tasks.

### 📄 Evaluation & Benchmarking

- 📄 [BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents](https://cdn.openai.com/pdf/5e10f4ab-d6f7-442e-9508-59515c65e35d/browsecomp.pdf) - Details on OpenAI's benchmark for browsing agents.

- 📄 [AI Agents That Matter](https://arxiv.org/abs/2407.01502) - Discusses shortcomings in agent benchmarks, proposing cost-controlled evaluations and standardization.

### 📄 Literature Review Systems

- 📄 [LitLLMs, LLMs for Literature Review: Are we there yet?](https://arxiv.org/abs/2412.15249) - Explores the potential of LLMs to assist in literature reviews, evaluating their current capabilities and limitations.

- 📄 [LitLLM: A Toolkit for Scientific Literature Review](https://arxiv.org/abs/2402.01788) - Presents a toolkit leveraging LLMs to help researchers identify, synthesize, and contextualize scientific publications.

- 📄 [PaperQA: Retrieval-Augmented Generative Agent for Scientific Research](https://arxiv.org/abs/2312.07559) - Describes an AI system for scientific literature search.

### 📄 Agent Systems & Collaboration

- 📄 [Multi-agent deep reinforcement learning: a survey](https://link.springer.com/article/10.1007/s10458-021-09525-8) - Comprehensive overview of multi-agent reinforcement learning approaches.

- 📄 [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) - Introduces multi-agent frameworks for collaboration.

- 📄 [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://arxiv.org/abs/2408.07199) - Proposes a framework combining MCTS with self-critique for better decision-making in dynamic environments.

---

## 🤝 Contributions

<div align="center">
  <p><i>Join us in building the most comprehensive resource on deep research tools and technologies</i></p>
</div>

We welcome contributions! If you know of a deep research tool, implementation, dataset, or resource that belongs on this list, feel free to submit a PR.

Please follow these guidelines:

- Create a new branch for your additions or edits (do not commit to main directly).
- Add your contribution – whether it's a research system, benchmark, or important paper. Please include a short description and relevant links (and ensure the format matches the existing entries).
- Submit a Pull Request (PR) – keep it focused and specific. For example, add one project or a few related items per PR, rather than sweeping changes. This makes review easier.

A repository maintainer will review your PR, provide feedback if needed, and merge it once approved. By contributing, you agree to abide by the repository's style and scope. Let's collaboratively maintain the best resource on deep research! 🚀

---

## 📜 License

<div align="center">
  <p><i>Legal information about usage and contribution rights</i></p>
</div>

This project is licensed under the MIT License. By contributing, you agree that your contributions will be also released under this license. For details, see the LICENSE file.

Happy researching with AI agents! 🚀