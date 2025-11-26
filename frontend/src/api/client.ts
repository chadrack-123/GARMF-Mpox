import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Studies
export const getStudies = () => api.get('/studies')
export const getStudy = (id: number) => api.get(`/studies/${id}`)
export const createStudy = (data: any) => api.post('/studies', data)
export const deleteStudy = (id: number) => api.delete(`/studies/${id}`)

// Datasets
export const getStudyDatasets = (studyId: number) => 
  api.get(`/datasets/studies/${studyId}/datasets`)
export const createDataset = (data: any) => api.post('/datasets', data)
export const getDataset = (id: number) => api.get(`/datasets/${id}`)

// Configs
export const getStudyConfigs = (studyId: number) => 
  api.get(`/configs/studies/${studyId}/configs`)
export const createConfig = (studyId: number, data: any) => 
  api.post(`/configs/studies/${studyId}/configs`, data)
export const getConfig = (id: number) => api.get(`/configs/${id}`)

// Runs
export const getRuns = () => api.get('/runs')
export const getRun = (id: number) => api.get(`/runs/${id}`)
export const createRun = (data: any) => api.post('/runs', data)

// Artefacts
export const getRunArtefacts = (runId: number) => 
  api.get(`/artefacts/runs/${runId}/artefacts`)
export const downloadArtefact = (id: number) => 
  api.get(`/artefacts/${id}/download`, { responseType: 'blob' })

// XAI
export const getRunXAI = (runId: number) => api.get(`/artefacts/runs/${runId}/xai`)

// Documentation
export const getRunDocs = (runId: number) => api.get(`/artefacts/runs/${runId}/docs`)

// Bundles
export const getBundle = (runId: number) => 
  api.get(`/artefacts/bundles/${runId}`, { responseType: 'blob' })

export default api
