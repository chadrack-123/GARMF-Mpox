import { Container, Typography } from '@mui/material'

export default function DatasetManager() {
  return (
    <Container maxWidth="xl">
      <Typography variant="h4" gutterBottom>
        Dataset Manager
      </Typography>
      <Typography variant="body1">
        Upload and manage datasets. Register new datasets, validate data contracts, and view schema information.
      </Typography>
    </Container>
  )
}
