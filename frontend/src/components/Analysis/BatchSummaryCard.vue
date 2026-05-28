<template>
  <el-card v-if="summary" class="batch-summary-card" shadow="never">
    <template #header>
      <div class="summary-header">
        <div>
          <h3>批量总体报告</h3>
          <p>{{ summary.title || '批量分析' }}</p>
        </div>
        <el-tag :type="statusType">{{ statusText }}</el-tag>
      </div>
    </template>

    <el-row :gutter="12" class="summary-stats">
      <el-col :span="6">
        <div class="stat-item">
          <div class="stat-value">{{ summary.total_tasks }}</div>
          <div class="stat-label">总数</div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="stat-item">
          <div class="stat-value">{{ summary.completed_tasks }}</div>
          <div class="stat-label">已完成</div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="stat-item">
          <div class="stat-value">{{ summary.failed_tasks }}</div>
          <div class="stat-label">失败</div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="stat-item">
          <div class="stat-value">{{ summary.pending_tasks }}</div>
          <div class="stat-label">待完成</div>
        </div>
      </el-col>
    </el-row>

    <div class="cost-panel">
      <div class="cost-item">
        <span class="cost-label">DeepSeek余额差额</span>
        <span class="cost-value">{{ formatOfficialCost }}</span>
      </div>
      <div class="cost-item">
        <span class="cost-label">本地估算</span>
        <span class="cost-value">{{ formatLocalCost }}</span>
      </div>
      <div v-if="costNotice" class="cost-notice">{{ costNotice }}</div>
    </div>

    <el-table :data="summary.items || []" size="small" style="width: 100%; margin-top: 16px;">
      <el-table-column prop="stock_code" label="股票代码" width="120" />
      <el-table-column prop="stock_name" label="股票名称" min-width="140">
        <template #default="{ row }">
          <el-button v-if="row.stock_code || row.symbol" link type="primary" @click="openStock(row)">
            {{ row.stock_name || row.stock_code || row.symbol }}
          </el-button>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column label="单项报告" width="110">
        <template #default="{ row }">
          <el-button
            v-if="row.report_available && row.report_id"
            link
            type="primary"
            @click="openReport(row)"
          >
            查看报告
          </el-button>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column label="状态" width="100">
        <template #default="{ row }">
          <el-tag size="small" :type="rowStatusType(row.status)">
            {{ rowStatusText(row.status) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="recommendation" label="投资建议" width="120">
        <template #default="{ row }">
          {{ row.recommendation || row.action_label || actionText(row.action) || '-' }}
        </template>
      </el-table-column>
      <el-table-column label="价格预测" min-width="220">
        <template #default="{ row }">
          <div v-if="row.price_prediction" class="price-prediction-list">
            <span>目标 {{ formatPrice(row.price_prediction.target_price) }}</span>
            <span>止损 {{ formatPrice(row.price_prediction.stop_loss_price) }}</span>
            <span>止盈 {{ formatPrice(row.price_prediction.take_profit_price) }}</span>
            <span>观察 {{ formatPrice(row.price_prediction.watch_price) }}</span>
          </div>
          <span v-else>{{ formatPrice(row.target_price) }}</span>
        </template>
      </el-table-column>
      <el-table-column prop="confidence" label="模型置信度" width="130">
        <template #default="{ row }">
          {{ formatConfidence(row.confidence) }}
        </template>
      </el-table-column>
      <el-table-column prop="error_message" label="失败原因" min-width="180">
        <template #default="{ row }">
          <span class="error-text">{{ row.error_message || dataStatusText(row) || '-' }}</span>
        </template>
      </el-table-column>
    </el-table>
  </el-card>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import type { BatchSummary } from '@/api/analysis'

const props = defineProps<{
  summary: BatchSummary | null
}>()
const router = useRouter()

const statusMap: Record<string, string> = {
  pending: '等待中',
  running: '处理中',
  processing: '处理中',
  completed: '已完成',
  partial_success: '部分成功',
  failed: '失败',
  cancelled: '已取消',
  submitted: '已提交'
}

const statusType = computed(() => {
  const status = props.summary?.status
  if (status === 'completed') return 'success'
  if (status === 'partial_success') return 'warning'
  if (status === 'failed') return 'danger'
  return 'info'
})

const statusText = computed(() => statusMap[props.summary?.status || ''] || props.summary?.status || '-')

const rowStatusType = (status: string): 'success' | 'warning' | 'danger' | 'info' => {
  if (status === 'completed') return 'success'
  if (status === 'failed') return 'danger'
  if (status === 'processing' || status === 'running') return 'warning'
  return 'info'
}

const rowStatusText = (status: string) => statusMap[status] || status || '-'

const actionText = (action?: string | null) => {
  if (action === 'BUY') return '买入'
  if (action === 'SELL') return '卖出'
  if (action === 'HOLD') return '持有'
  return ''
}

const dataStatusText = (row: any) => {
  if (row?.data_status === 'data_missing') return row.reasoning || '缺少行情/估值数据'
  return ''
}

const formatPrice = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return `${Number(value).toFixed(2)}`
}

const formatConfidence = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  const numeric = Number(value)
  return numeric <= 1 ? `${(numeric * 100).toFixed(1)}%` : `${numeric.toFixed(1)}%`
}

const formatCurrencyMap = (value?: Record<string, number>) => {
  const entries = Object.entries(value || {}).filter(([, amount]) => Number.isFinite(Number(amount)))
  if (!entries.length) return '-'
  return entries
    .map(([currency, amount]) => `${currency === 'CNY' ? '¥' : `${currency} `}${Number(amount).toFixed(4)}`)
    .join(' / ')
}

const formatOfficialCost = computed(() => {
  const cost = props.summary?.cost_summary
  if (!cost) return '未获取'
  if (!cost.official_available) return cost.official_reason || '未获取'
  return formatCurrencyMap(cost.official_cost_delta_by_currency)
})

const formatLocalCost = computed(() => {
  const cost = props.summary?.cost_summary
  if (!cost) return '未获取'
  if (!cost.local_estimate_available) return cost.local_estimate_reason || '未获取'
  return formatCurrencyMap(cost.local_estimated_cost_by_currency)
})

const costNotice = computed(() => {
  const cost = props.summary?.cost_summary
  if (!cost) return ''
  return cost.balance_warning || ''
})

const openStock = (row: any) => {
  const code = row.stock_code || row.symbol
  if (!code) return
  router.push({ name: 'StockDetail', params: { code } })
}

const openReport = (row: any) => {
  if (!row.report_id) return
  router.push({ name: 'ReportDetail', params: { id: row.report_id } })
}
</script>

<style scoped lang="scss">
.batch-summary-card {
  margin-top: 16px;

  .summary-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;

    h3 {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
    }

    p {
      margin: 4px 0 0;
      color: var(--el-text-color-secondary);
      font-size: 13px;
    }
  }

  .summary-stats {
    .stat-item {
      border: 1px solid var(--el-border-color-light);
      border-radius: 6px;
      padding: 12px;
      text-align: center;
      background: var(--el-fill-color-lighter);
    }

    .stat-value {
      font-size: 22px;
      font-weight: 700;
      line-height: 1.2;
    }

    .stat-label {
      color: var(--el-text-color-secondary);
      font-size: 12px;
      margin-top: 4px;
    }
  }

  .cost-panel {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    border: 1px solid var(--el-border-color-light);
    border-radius: 6px;
    padding: 12px;
    margin-top: 16px;
    background: var(--el-fill-color-blank);
  }

  .cost-item {
    display: flex;
    align-items: baseline;
    gap: 8px;
  }

  .cost-label {
    color: var(--el-text-color-secondary);
    font-size: 12px;
  }

  .cost-value {
    color: var(--el-text-color-primary);
    font-size: 14px;
    font-weight: 600;
  }

  .cost-notice {
    color: var(--el-color-warning);
    font-size: 12px;
  }

  .error-text {
    color: var(--el-color-danger);
  }

  .price-prediction-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 10px;
    font-size: 12px;
    line-height: 1.5;
  }
}
</style>
