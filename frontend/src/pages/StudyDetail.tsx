import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Box,
  Container,
  Tab,
  Tabs,
  Typography,
  Paper,
} from '@mui/material'
import { useState } from 'react'
import { getStudy, getStudyDatasets, getStudyConfigs } from '../api/client'

type Study = {
  id: number
  title: string
  disease: string
  modality: string
  doi?: string
  citation?: string
}

type Dataset = {
  id: number
  type: string
  storage_uri: string
}

type Config = {
  id: number
  name: string
  config_type: string
}

export default function StudyDetail() {
  const { studyId } = useParams<{ studyId: string }>()
  const [tab, setTab] = useState(0)

  // Safely handle studyId possibly being undefined
  const numericStudyId = studyId ? Number(studyId) : undefined

  const { data: study } = useQuery<Study>({
    queryKey: ['study', numericStudyId],
    queryFn: async () => {
      if (!numericStudyId) {
        throw new Error('No studyId provided')
      }
      const response = await getStudy(numericStudyId)
      return response.data as Study
    },
    enabled: !!numericStudyId,
  })

  const { data: datasets } = useQuery<Dataset[]>({
    queryKey: ['datasets', numericStudyId],
    queryFn: async () => {
      if (!numericStudyId) {
        throw new Error('No studyId provided')
      }
      const response = await getStudyDatasets(numericStudyId)
      return response.data as Dataset[]
    },
    enabled: !!numericStudyId,
  })

  const { data: configs } = useQuery<Config[]>({
    queryKey: ['configs', numericStudyId],
    queryFn: async () => {
      if (!numericStudyId) {
        throw new Error('No studyId provided')
      }
      const response = await getStudyConfigs(numericStudyId)
      return response.data as Config[]
    },
    enabled: !!numericStudyId,
  })

  return (
    <Container maxWidth="xl">
      <Typography variant="h4" gutterBottom>
        {study?.title}
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs
          value={tab}
          onChange={(_event, newValue: number) => setTab(newValue)}
        >
          <Tab label="Overview" />
          <Tab label="Datasets" />
          <Tab label="Configs" />
          <Tab label="Runs" />
        </Tabs>
      </Box>

      {tab === 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Study Information
          </Typography>
          <Typography><strong>Disease:</strong> {study?.disease}</Typography>
          <Typography><strong>Modality:</strong> {study?.modality}</Typography>
          {study?.doi && (
            <Typography><strong>DOI:</strong> {study.doi}</Typography>
          )}
          {study?.citation && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2">Citation:</Typography>
              <Typography variant="body2">{study.citation}</Typography>
            </Box>
          )}
        </Paper>
      )}

      {tab === 1 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Datasets ({datasets?.length ?? 0})
          </Typography>
          {datasets?.map((dataset) => (
            <Box
              key={dataset.id}
              sx={{ mb: 2, p: 2, border: '1px solid #ddd', borderRadius: 1 }}
            >
              <Typography><strong>Type:</strong> {dataset.type}</Typography>
              <Typography><strong>URI:</strong> {dataset.storage_uri}</Typography>
            </Box>
          ))}
        </Paper>
      )}

      {tab === 2 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Configurations ({configs?.length ?? 0})
          </Typography>
          {configs?.map((config) => (
            <Box
              key={config.id}
              sx={{ mb: 2, p: 2, border: '1px solid #ddd', borderRadius: 1 }}
            >
              <Typography><strong>Name:</strong> {config.name}</Typography>
              <Typography><strong>Type:</strong> {config.config_type}</Typography>
            </Box>
          ))}
        </Paper>
      )}
    </Container>
  )
}
