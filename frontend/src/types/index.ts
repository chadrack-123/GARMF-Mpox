export interface Study {
  id: number
  title: string
  citation?: string
  doi?: string
  disease: string
  modality: 'image' | 'tabular' | 'mixed'
  notes?: string
}

export interface Dataset {
  id: number
  study_id: number
  type: 'image' | 'tabular'
  storage_uri: string
  checksum?: string
  data_contract?: any
  license?: string
}

export interface Config {
  id: number
  study_id: number
  name: string
  yaml_text: string
  created_at: string
  config_type: 'baseline' | 'framework_enhanced'
}

export interface Run {
  id: number
  study_id: number
  config_id: number
  kind: 'baseline' | 'framework_enhanced'
  status: 'pending' | 'running' | 'completed' | 'failed'
  started_at?: string
  finished_at?: string
  metrics?: Record<string, any>
  reproducibility_metrics?: Record<string, any>
  environment_info?: Record<string, any>
  random_seeds?: Record<string, number>
  split_hash?: string
  container_digest?: string
  error_message?: string
}

export interface Artefact {
  id: number
  run_id: number
  type: 'log' | 'model' | 'metric_plot' | 'xai' | 'doc' | 'bundle'
  path: string
  metadata?: Record<string, any>
}
