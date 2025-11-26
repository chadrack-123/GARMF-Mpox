import { useParams } from 'react-router-dom'
import { Container, Typography } from '@mui/material'

export default function DocumentationViewer() {
  const { runId } = useParams<{ runId: string }>()

  return (
    <Container maxWidth="xl">
      <Typography variant="h4" gutterBottom>
        Documentation - Run {runId}
      </Typography>
      <Typography variant="body1">
        View data cards, model cards, and reproducibility checklists.
      </Typography>
    </Container>
  )
}
