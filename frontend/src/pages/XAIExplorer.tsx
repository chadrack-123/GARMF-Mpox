import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Container, Typography, Grid, Paper, Box } from '@mui/material'
import { getRunXAI } from '../api/client'

export default function XAIExplorer() {
  const { runId } = useParams<{ runId: string }>()

  type XAIArtefact = {
    id: number
    path: string
    metadata?: {
      type?: string
      [key: string]: any
    }
    [key: string]: any
  }

  const { data: xaiArtefacts } = useQuery<XAIArtefact[]>({
    queryKey: ['xai', runId],
    queryFn: async () => {
      const response = await getRunXAI(Number(runId))
      return response.data as XAIArtefact[]
    },
  })

  return (
    <Container maxWidth="xl">
      <Typography variant="h4" gutterBottom>
        XAI Explorer - Run {runId}
      </Typography>

      <Grid container spacing={3}>
        {xaiArtefacts?.map((artefact: any) => (
          <Grid item xs={12} md={6} key={artefact.id}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                {artefact.metadata?.type || 'XAI Output'}
              </Typography>
              <Box sx={{ 
                border: '1px solid #ddd', 
                borderRadius: 1, 
                p: 2,
                textAlign: 'center' 
              }}>
                <Typography variant="body2" color="text.secondary">
                  {artefact.path}
                </Typography>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Container>
  )
}
