import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Base Model Lab',
  description: '从零训练一个迷你基座模型（Mini Transformer Base Model）：动手实验与原理剖析',

  themeConfig: {
    siteTitle: '🧪 Base Model Lab',
    logo: undefined,
    nav: [
      { text: '首页', link: '/' },
      { text: '项目指南', link: '/guide/intro' },
      { text: '架构原理', link: '/architecture/transformer' },
      { text: '源码解析', link: '/code/train' },
      { text: '实验指标', link: '/experiment/benchmark' },
      { text: '进阶演进', link: '/advanced/scaling' }
    ],

    sidebar: [
      {
        text: '📖 项目指南',
        collapsed: false,
        items: [
          { text: '项目简介与定位', link: '/guide/intro' },
          { text: '快速上手复现', link: '/guide/quickstart' },
          { text: '终端交互体验', link: '/guide/chat' }
        ]
      },
      {
        text: '🧠 核心原理',
        collapsed: false,
        items: [
          { text: '什么是基座模型？', link: '/architecture/what-is-base-model' },
          { text: 'Transformer 结构与因果自注意力', link: '/architecture/transformer' },
          { text: '预训练：Next-Token Prediction', link: '/architecture/pretraining' }
        ]
      },
      {
        text: '💻 源码深度拆解',
        collapsed: false,
        items: [
          { text: '字符级分词器 (CharTokenizer)', link: '/code/tokenizer' },
          { text: '模型架构实现 (TinyTransformer)', link: '/code/train' },
          { text: '训练循环与温度采样 (generate)', link: '/code/training-and-generation' },
          { text: '实验工具与因果掩码单测', link: '/code/experiments-and-tests' }
        ]
      },
      {
        text: '📊 实验分析与评测',
        collapsed: false,
        items: [
          { text: '实测数据与损失曲线', link: '/experiment/benchmark' },
          { text: '局限性与边界分析', link: '/experiment/limitations' }
        ]
      },
      {
        text: '🚀 进阶与工业级演进',
        collapsed: false,
        items: [
          { text: '从玩具模型到万亿参数', link: '/advanced/scaling' }
        ]
      }
    ],

    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭'
            }
          }
        }
      }
    },

    footer: {
      message: '基于 MIT 协议开放 · 从零理解大语言模型训练本质',
      copyright: 'Copyright © 2026 Base Model Lab'
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    lastUpdated: {
      text: '最后更新于'
    }
  }
})
